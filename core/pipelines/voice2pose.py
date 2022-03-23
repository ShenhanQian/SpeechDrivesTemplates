import logging
import os
import time
from collections import OrderedDict
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn import decomposition

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel.data_parallel import DataParallel
import torchaudio

from .trainer import Trainer
from core.networks import get_model
from core.utils.keypoint_visualization import vis_relative_pose_pair_clip, vis_relative_pose_clip, draw_pose_frames_in_long_img
from core.utils.fgd import compute_fgd


class Voice2PoseModel(nn.Module):
    def __init__(self, cfg, state_dict=None, num_train_samples=None, rank=0) -> None:
        super().__init__()
        self.cfg = cfg

        self.mel_transfm = torchaudio.transforms.MelSpectrogram(
            win_length=400, hop_length=160,
            n_fft=512, f_min=55,
            f_max=7500.0, n_mels=80)
        
        # Generator
        self.netG = get_model(cfg.VOICE2POSE.GENERATOR.NAME)(cfg)

        ## regression loss
        self.reg_criterion = nn.L1Loss(reduction='none')

        if cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:

            if cfg.VOICE2POSE.GENERATOR.CLIP_CODE.EXTERNAL_CODE:
                if cfg.VOICE2POSE.GENERATOR.CLIP_CODE.EXTERNAL_CODE_PTH is not None:
                    map_location = {'cuda:0' : 'cuda:%d' % rank}
                    ckpt = torch.load(cfg.VOICE2POSE.GENERATOR.CLIP_CODE.EXTERNAL_CODE_PTH, map_location=map_location)
                elif cfg.VOICE2POSE.POSE_ENCODER.AE_CHECKPOINT is not None:
                    map_location = {'cuda:0' : 'cuda:%d' % rank}
                    ckpt = torch.load(cfg.VOICE2POSE.POSE_ENCODER.AE_CHECKPOINT, map_location=map_location)
                else:
                    raise RuntimeError('External code not provide.')
                
                self.clips_code = dict(
                    map(lambda x: (x[0].replace('module.', ''), x[1]), 
                        filter(lambda x: 'clip_code' in x[0],
                            ckpt['model_state_dict'].items())
                        )
                    )['clip_code_mu']
                # assert self.clips_code.shape[0] == num_train_samples, \
                #     'Mismatched external code from %s' % cfg.VOICE2POSE.POSE_ENCODER.AE_CHECKPOINT

            else:
                if num_train_samples is None:
                    assert state_dict is not None, 'No state_dict available, while no dataset is configured.'
                    num_train_samples = state_dict['module.clips_code'].shape[0]
                self.clips_code = (torch.zeros([1, cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION])
                    ).repeat((num_train_samples, 1))
                
                if self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.FRAME_VARIANT:
                    self.clips_code = self.clips_code[..., None].repeat([1, 1, self.cfg.DATASET.NUM_FRAMES])

                self.clips_code =  nn.Parameter(self.clips_code,
                    requires_grad=self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.TRAIN)
        else:
            self.clips_code = None
        
        # pose encoder
        if self.cfg.VOICE2POSE.POSE_ENCODER.NAME is not None:
            self.pose_encoder = get_model(cfg.VOICE2POSE.POSE_ENCODER.NAME)(cfg)
            self.pose_encoder.eval()

        # Pose Discriminator
        if self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.NAME is not None:
            self.netD_pose = get_model(cfg.VOICE2POSE.POSE_DISCRIMINATOR.NAME)(cfg)
            self.pose_gan_criterion = nn.MSELoss()
        
    def forward(self, batch, dataset, return_loss=True, interpolation_coeff=None):
        # input
        audio = batch['audio'].cuda()
        speaker = batch['speaker']
        clip_indices = batch['clip_index'].cuda()
        num_frames = int(batch['num_frames'][0].item())
        poses_gt_batch = batch['poses'].cuda() if return_loss else None

        if self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:
            if self.training:
                condition_code = self.clips_code[clip_indices].cuda()
            else:
                if self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.SAMPLE_FROM_NORMAL:
                    condition_code = torch.randn(
                        [len(clip_indices), self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION]
                        ).cuda()
                elif self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.TEST_WITH_GT_CODE:
                    assert self.cfg.VOICE2POSE.POSE_ENCODER.NAME is not None
                    if self.cfg.DATASET.HIERARCHICAL_POSE:
                        mu_gt, logvar_gt = self.pose_encoder(poses_gt_batch)
                    else:
                        mu_gt, logvar_gt = self.pose_encoder(dataset.transform_normalized_parted2global(poses_gt_batch, speaker))
                    condition_code = mu_gt
                elif self.cfg.DEMO.CODE_INDEX is not None:
                    assert not return_loss, 'WARNING: Do not set "DEMO.CODE_INDEX" in train or test mode!'
                    assert 0 <= self.cfg.DEMO.CODE_INDEX < self.clips_code.size(0)

                    indices = torch.ones((len(audio),)).long() * self.cfg.DEMO.CODE_INDEX
                    condition_code =  self.clips_code[indices].cuda()
                    if interpolation_coeff is not None:
                        assert self.cfg.DEMO.CODE_INDEX_B < self.clips_code.size(0)
                        indices_b = torch.ones((len(audio),)).long() * self.cfg.DEMO.CODE_INDEX_B
                        condition_code_b =  self.clips_code[indices_b].cuda()
                        condition_code = condition_code * (1-interpolation_coeff) + condition_code_b * interpolation_coeff
                else:
                    rand_indices = torch.randint(self.clips_code.size(0), (len(audio),))
                    condition_code =  self.clips_code[rand_indices].cuda()
        else: 
            condition_code = None

        # forward
        mel = self.mel_transfm(audio)
        poses_pred_batch = self.netG(mel, num_frames, condition_code)
        
        results_dict = {
            'poses_pred_batch': poses_pred_batch,
            'condition_code': condition_code,
            }
        if not return_loss:
            return results_dict
        else:
            results_dict['poses_gt_batch'] = poses_gt_batch

        losses_dict = {}

        ## netG
        ### regression loss
        G_reg_loss = self.reg_criterion(poses_pred_batch, poses_gt_batch) * self.cfg.VOICE2POSE.GENERATOR.LAMBDA_REG
        G_reg_loss = G_reg_loss.mean()
        losses_dict['G_reg_loss'] = G_reg_loss
        G_loss = G_reg_loss.clone()

        ### ClipCode regularization
        if condition_code is not None:
            if self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.FRAME_VARIANT:
                clipcode_mu = condition_code.permute([0, 2, 1]).reshape(-1, self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION).mean(dim=0)
                clipcode_var = condition_code.permute([0, 2, 1]).reshape(-1, self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION).var(dim=0)
            else:
                clipcode_mu = condition_code.mean(dim=0)
                clipcode_var = condition_code.var(dim=0)
            if (clipcode_var!=0).all():
                G_clipcode_kl_loss = 0.5 * (-torch.log(clipcode_var) + clipcode_mu**2 + clipcode_var - 1).mean() * self.cfg.VOICE2POSE.GENERATOR.LAMBDA_CLIP_KL
                losses_dict['G_clipcode_kl_loss'] = G_clipcode_kl_loss
                G_loss = G_loss + G_clipcode_kl_loss

        losses_dict['G_loss'] = G_loss
        
        ## pose encoder
        if self.cfg.VOICE2POSE.POSE_ENCODER.NAME is not None:
            with torch.no_grad():
                if self.cfg.DATASET.HIERARCHICAL_POSE:
                    mu_pred, logvar_pred = self.pose_encoder(poses_pred_batch)
                    mu_gt, logvar_gt = self.pose_encoder(poses_gt_batch)
                else:
                    mu_pred, logvar_pred = self.pose_encoder(dataset.transform_normalized_parted2global(poses_pred_batch, speaker))
                    mu_gt, logvar_gt = self.pose_encoder(dataset.transform_normalized_parted2global(poses_gt_batch, speaker))

                results_dict.update({
                    'mu_pred': mu_pred,
                    'mu_gt': mu_gt,
                    'logvar_pred': logvar_pred,
                    'logvar_gt': logvar_gt,
                    })

        ## netD_pose
        if hasattr(self, 'netD_pose'):
            real_batch = poses_gt_batch
            fake_batch = poses_pred_batch
            if self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.WHITE_LIST is not None:
                white_list = self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.WHITE_LIST
                real_batch = real_batch[..., white_list]
                fake_batch = fake_batch[..., white_list]

            if self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.MOTION:
                real_batch = real_batch[:, 1:, ...] - real_batch[:, :-1, ...]
                fake_batch = fake_batch[:, 1:, ...] - fake_batch[:, :-1, ...]
                
            pose_score_real = self.netD_pose(real_batch)
            pose_score_fake = self.netD_pose(fake_batch)
            pose_score_fake_deatchG = self.netD_pose(fake_batch.detach())
            
            G_pose_gan_loss = self.pose_gan_criterion(pose_score_fake, torch.ones_like(pose_score_fake)) * self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.LAMBDA_GAN
            losses_dict['G_pose_gan_loss'] = G_pose_gan_loss
            G_loss = G_loss + G_pose_gan_loss
            losses_dict['G_loss'] = G_loss

            D_pose_gan_fake_loss = self.pose_gan_criterion(pose_score_fake_deatchG, torch.zeros_like(pose_score_fake_deatchG))
            D_pose_gan_real_loss = self.pose_gan_criterion(pose_score_real, torch.ones_like(pose_score_real))
            D_pose_gan_loss = (D_pose_gan_real_loss + D_pose_gan_fake_loss) * self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.LAMBDA_GAN

            losses_dict.update({
                'D_pose_gan_loss': D_pose_gan_loss,
                'pose_score_fake': pose_score_fake.mean(),
                'pose_score_real': pose_score_real.mean(),
                })
        
        return losses_dict, results_dict

class Voice2Pose(Trainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
    
    def setup_model(self, cfg, state_dict=None):
        if self.is_master_process():
            print(torch.cuda.device_count(), "GPUs are available.")
        print('Setting up models on rank', self.get_rank())

        self.model = Voice2PoseModel(cfg, state_dict, self.num_train_samples, self.get_rank()).cuda()
        if self.cfg.SYS.DISTRIBUTED:
            self.model = DDP(self.model, device_ids=[self.get_rank()], find_unused_parameters=True)
        else:
            self.model = DataParallel(self.model)

        if state_dict is not None:
            if self.cfg.VOICE2POSE.STRICT_LOADING:
                self.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict, strict=False)
        
        # pose encoder
        if self.cfg.VOICE2POSE.POSE_ENCODER.NAME is not None:
            if cfg.VOICE2POSE.POSE_ENCODER.AE_CHECKPOINT is not None:
                map_location = {'cuda:0' : 'cuda:%d' % self.get_rank()}
                ckpt = torch.load(cfg.VOICE2POSE.POSE_ENCODER.AE_CHECKPOINT, map_location=map_location)
                state_dict = OrderedDict(
                    map(lambda x: (x[0].replace('module.ae.encoder.', ''), x[1]), 
                        filter(lambda x: 'encoder' in x[0],
                            ckpt['model_state_dict'].items())))
                self.model.module.pose_encoder.load_state_dict(state_dict)

    def setup_optimizer(self, checkpoint=None, last_epoch=-1):
        # Optimizer for generator
        netG_parameters = self.model.module.netG.parameters() if isinstance(self.model, (DDP, DataParallel)) \
            else self.model.netG.parameters()
        
        self.optimizers['optimizerG'] = torch.optim.Adam(netG_parameters, lr=self.cfg.TRAIN.LR,
                                                         weight_decay=self.cfg.TRAIN.WD)
        if checkpoint is not None:
            self.optimizers['optimizerG'].load_state_dict(checkpoint['optimizerG_state_dict'])
        if self.cfg.TRAIN.LR_SCHEDULER:
            self.schedulers['schedulerG'] = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizers['optimizerG'],
                [self.cfg.TRAIN.NUM_EPOCHS-10, self.cfg.TRAIN.NUM_EPOCHS-2],
                gamma=0.1, last_epoch=last_epoch)

        # Optimizer for pose discriminator
        if self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.NAME is not None:
            netD_pose_parameters = self.model.module.netD_pose.parameters() if isinstance(self.model, (DDP, DataParallel)) \
                else self.model.netD_pose.parameters()
            self.optimizers['optimizerD_pose'] = torch.optim.Adam(netD_pose_parameters, lr=self.cfg.TRAIN.LR)
            if checkpoint is not None:
                self.optimizers['optimizerD_pose'].load_state_dict(checkpoint['optimizerD_pose_state_dict'])
            if self.cfg.TRAIN.LR_SCHEDULER:
                self.schedulers['schedulerD_pose'] = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizers['optimizerD_pose'], [self.cfg.TRAIN.NUM_EPOCHS-10, self.cfg.TRAIN.NUM_EPOCHS-2], gamma=0.1, last_epoch=last_epoch)
        
        # Optimizer for clip code
        if self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None and not self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.EXTERNAL_CODE:
            code_parameters = [self.model.module.clips_code] if isinstance(self.model, (DDP, DataParallel)) \
                else [self.model.clips_code]
            self.optimizers['optimizerClipCode'] = torch.optim.Adam(code_parameters, lr=self.cfg.TRAIN.LR * self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.LR_SCALING)
            if checkpoint is not None:
                self.optimizers['optimizerClipCode'].load_state_dict(checkpoint['optimizerClipCode_state_dict'])
            if self.cfg.TRAIN.LR_SCHEDULER:
                self.schedulers['schedulerClipCode'] = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizers['optimizerClipCode'], [self.cfg.TRAIN.NUM_EPOCHS-10, self.cfg.TRAIN.NUM_EPOCHS-2], gamma=0.1, last_epoch=last_epoch)

    def train_step(self, batch, t_step, global_step, epoch):
        tag = 'TRAIN'
        dataset = self.train_dataset

        audio = batch['audio']
        speaker_stat = batch['speaker_stat']

        losses_dict, results_dict = self.model(batch, dataset)
        results_dict['poses_pred_batch'] = dataset.get_final_results(results_dict['poses_pred_batch'].detach(), speaker_stat)
        results_dict['poses_gt_batch'] = dataset.get_final_results(results_dict['poses_gt_batch'].detach(), speaker_stat)
        
        losses_dict.update(self.evaluate_step(results_dict))

        if not self.cfg.SYS.DISTRIBUTED:
            losses_dict = dict(map(lambda x: (x[0], x[1].mean()), losses_dict.items()))
        
        # optimization
        if 'optimizerClipCode' in self.optimizers:
            self.optimizers['optimizerClipCode'].zero_grad()
        self.optimizers['optimizerG'].zero_grad()
        losses_dict['G_loss'].backward(retain_graph=True)
        if 'optimizerClipCode' in self.optimizers:
            self.optimizers['optimizerClipCode'].step()
        self.optimizers['optimizerG'].step()

        if 'optimizerD_pose' in self.optimizers:
            self.optimizers['optimizerD_pose'].zero_grad()
            losses_dict['D_pose_gan_loss'].backward()
            self.optimizers['optimizerD_pose'].step()
        
        if self.cfg.SYS.DISTRIBUTED:
            self.reduce_tensor_dict(losses_dict)

        if self.is_master_process():
            if t_step % self.cfg.SYS.LOG_INTERVAL == 0:
                self.logger_writer_step(tag, losses_dict, t_step, epoch, global_step)

            if t_step % (self.result_saving_interval_train) == 0:
                results_dict = dict(
                    map(lambda x: (x[0], x[1].detach().cpu().numpy()),
                        filter(lambda x: x[1] is not None, 
                            results_dict.items())))
                if self.cfg.TRAIN.SAVE_NPZ:
                    self.save_results(tag, t_step, epoch, self.base_path, results_dict)
                if self.cfg.TRAIN.SAVE_VIDEO:
                    relative_poses_pred = results_dict['poses_pred_batch'][0]
                    relative_poses_gt = results_dict['poses_gt_batch'][0]
                    vid_batch = self.generate_video_pair(relative_poses_pred, relative_poses_gt)
                    self.video_writer.save_video(
                        self.cfg, tag, vid_batch, t_step, epoch, global_step,
                        audio=audio[0].numpy(), writer=self.tb_writer, base_path=self.base_path)

    def test_step(self, batch, t_step, epoch=0):
        tag = 'TEST' if epoch == 0 else 'VAL'
        dataset = self.test_dataset

        # multiple test
        assert isinstance(self.cfg.TEST.MULTIPLE, int) and self.cfg.TEST.MULTIPLE >= 1, \
            f'TEST.MULTIPLE should be an integer that larger than 1, ' \
            + f'but get {self.cfg.TEST.MULTIPLE} ({type(self.cfg.TEST.MULTIPLE)}).'
        if self.cfg.TEST.MULTIPLE > 1:
            batch = self.mutiply_batch(batch, self.cfg.TEST.MULTIPLE)
        
        audio = batch['audio']
        speaker_stat = batch['speaker_stat']

        losses_dict, results_dict = self.model(batch, dataset)
        results_dict['poses_pred_batch'] = dataset.get_final_results(results_dict['poses_pred_batch'].detach(), speaker_stat)
        results_dict['poses_gt_batch'] = dataset.get_final_results(results_dict['poses_gt_batch'].detach(), speaker_stat)

        losses_dict.update(self.evaluate_step(results_dict))

        if not self.cfg.SYS.DISTRIBUTED:
            losses_dict = dict(map(lambda x: (x[0], x[1].mean()), losses_dict.items()))
        if self.cfg.SYS.DISTRIBUTED:
            self.reduce_tensor_dict(losses_dict)

        results_dict = dict(
            map(lambda x: (x[0], x[1].detach().cpu().numpy()),
                filter(lambda x: x[1] is not None, 
                    results_dict.items())))

        if self.is_master_process():
            if t_step % self.cfg.SYS.LOG_INTERVAL == 0 and self.get_rank() == 0:
                self.logger_writer_step(tag, losses_dict, t_step, epoch)
            
            if t_step % (self.result_saving_interval_test) == 0:
                
                if self.cfg.TEST.SAVE_NPZ:
                    self.save_results(
                        tag, t_step, epoch, self.base_path, results_dict)
                if self.cfg.TEST.SAVE_VIDEO:
                    relative_poses_pred = results_dict['poses_pred_batch'][0]
                    relative_poses_gt = results_dict['poses_gt_batch'][0]
                    vid_batch = self.generate_video_pair(relative_poses_pred, relative_poses_gt)
                    self.video_writer.save_video(
                        self.cfg, tag, vid_batch, t_step, epoch, audio=audio[0].numpy(),
                        writer=self.tb_writer, base_path=self.base_path)
        
        batch_losses_dict = dict(map(lambda x: (x[0], x[1].detach() * self.cfg.TEST.BATCH_SIZE), losses_dict.items()))
        batch_results_dict = dict(
            filter(lambda x: x[0] in ['mu_pred', 'mu_gt', 'logvar_pred', 'logvar_gt', 'condition_code'],
                results_dict.items()))
        return batch_losses_dict, batch_results_dict
    
    def demo_step(self, batch, t_step, epoch=0, extra_id=None, interpolation_coeff=None):
        tag = 'DEMO'
        dataset = self.test_dataset

        audio = batch['audio']
        speaker_stat = batch['speaker_stat']

        results_dict = self.model(batch, dataset, return_loss=False, interpolation_coeff=interpolation_coeff)
        results_dict['poses_pred_batch'] = dataset.get_final_results(results_dict['poses_pred_batch'].detach(), speaker_stat)
        
        if self.is_master_process():
            results_dict = dict(
                map(lambda x: (x[0], x[1].detach().cpu().numpy()),
                    filter(lambda x: x[1] is not None, 
                        results_dict.items())))
            if self.cfg.TEST.SAVE_NPZ:
                self.save_results(
                    tag, t_step, epoch, self.base_path, results_dict, extra_id=extra_id)
            if self.cfg.TEST.SAVE_VIDEO:
                relative_poses_pred = results_dict['poses_pred_batch'][0]
                vid_batch = self.generate_video(relative_poses_pred)
                long_img = draw_pose_frames_in_long_img(relative_poses_pred.transpose(0,2,1))
                self.video_writer.save_video(
                    self.cfg, tag, vid_batch, t_step, epoch, long_img=long_img, audio=audio[0].numpy(),
                    writer=self.tb_writer, base_path=self.base_path, extra_id=extra_id)
    
    def evaluate_step(self, results_dict):
        poses_pred_batch = results_dict['poses_pred_batch']
        poses_gt_batch = results_dict['poses_gt_batch']
        
        L2_dist = torch.norm(poses_pred_batch-poses_gt_batch, p=2, dim=2)

        # lip sync error
        lip_open_pred = torch.norm(poses_pred_batch[:, :, :, 75] - poses_pred_batch[:, :, :, 71], p=2, dim=-1)
        lip_open_gt = torch.norm(poses_gt_batch[:, :, :, 75] - poses_gt_batch[:, :, :, 71], p=2, dim=-1)
        # normalized lip error
        lip_open_pred_n = lip_open_pred / (lip_open_gt.max(-1, keepdim=True).values + 1e-4)
        lip_open_gt_n = lip_open_gt / (lip_open_gt.max(-1, keepdim=True).values + 1e-4)
        lip_sync_error_n = torch.abs(lip_open_pred_n - lip_open_gt_n)
        
        metrics_dict = {
            'L2_dist': L2_dist.mean(),
            'lip_sync_error_n': lip_sync_error_n.mean(),
        }
        return metrics_dict
    
    def evaluate_epoch(self, results_dict):
        tic = time.time()
        metrics_dict = {}
        
        FGD_mu = compute_fgd(results_dict['mu_pred'], results_dict['mu_gt'])
        metrics_dict['FGD_mu'] = FGD_mu

        FGD_mu_logvar = compute_fgd(
            np.concatenate([results_dict['mu_pred'], results_dict['logvar_pred']], axis=1),
            np.concatenate([results_dict['mu_gt'], results_dict['logvar_gt']], axis=1))
        metrics_dict['FGD_mu_logvar'] = FGD_mu_logvar

        toc = time.time() - tic
        logging.info('Compelte epoch evaluation in %.2f min' % (toc/60))
        return metrics_dict

    def generate_video_pair(self, relative_poses_pred, relative_poses_gt):
        vid_batch = vis_relative_pose_pair_clip(
            relative_poses_pred * self.cfg.SYS.VISUALIZATION_SCALING,
            relative_poses_gt * self.cfg.SYS.VISUALIZATION_SCALING,
            self.cfg.SYS.CANVAS_SIZE)
        return vid_batch
    
    def generate_video(self, relative_poses):
        vid_batch = vis_relative_pose_clip(
            relative_poses * self.cfg.SYS.VISUALIZATION_SCALING,
            self.cfg.SYS.CANVAS_SIZE)
        return vid_batch
    
    def save_results(self, tag, step, epoch, base_path, results_dict, extra_id=None):
        res_tic = time.time()

        res_dir = os.path.join(base_path, 'results')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        pred_npz_path = '%s/epoch%d-%s-step%s.npz' %(res_dir, epoch, tag, step) \
            if extra_id is None \
            else '%s/epoch%d-%s-step%s-%d.npz' %(res_dir, epoch, tag, step, extra_id)
        if os.path.exists(pred_npz_path):
            os.remove(pred_npz_path)
        np.savez(pred_npz_path, **results_dict)

        res_toc = (time.time() - res_tic)
        logging.info('[%s] epoch: %d/%d  step: %d  Saved results in an %s file in %.3f seconds.' % (
            tag, epoch, self.cfg.TRAIN.NUM_EPOCHS, step, 'npz', res_toc))
    
    def draw_figure_epoch(self):
        fig_dict = {}
        mpl.use('Agg')
        mpl.rcParams['agg.path.chunksize'] = 10000
        kwargs = {}

        msg = '[TRAIN] epoch plotting: '

        if self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:
            assert self.model.module.clips_code is not None
            code = self.model.module.clips_code.detach().cpu().numpy()

            # kwargs = {'figsize': [6.4*2.5, 4.8*10], 'dpi': 100}
            kwargs = {}
            fig = plt.figure(**kwargs)

            pca = decomposition.PCA(n_components=2)
            if code.ndim == 3:
                code = code.reshape(-1, code.shape[-1])
            pca.fit(code)
            X = pca.transform(code)
            plt.scatter(X[:, 0], X[:, 1], alpha=0.2, edgecolors='none', s=1)

            fig.tight_layout()

            fig_dict['clip_code'] = fig
            plt.close()
            msg += 'Clip Code, '
        
        logging.info(msg)

        return fig_dict
    