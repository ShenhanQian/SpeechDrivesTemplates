import logging
import os
import time
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


class Pose2PoseModel(nn.Module):
    def __init__(self, cfg, state_dict=None, num_train_samples=None, rank=0) -> None:
        super().__init__()
        self.cfg = cfg

        self.mel_transfm = torchaudio.transforms.MelSpectrogram(
            win_length=400, hop_length=160,
            n_fft=512, f_min=55,
            f_max=7500.0, n_mels=80)

        # autoencoder
        self.ae = get_model(cfg.POSE2POSE.AUTOENCODER.NAME)(cfg)
        if num_train_samples is None:
            assert state_dict is not None, 'No state_dict available, while no dataset is configured.'
            num_train_samples = state_dict['module.clip_code_mu'].shape[0]
        self.register_buffer('clip_code_mu', torch.zeros([num_train_samples, cfg.POSE2POSE.AUTOENCODER.CODE_DIM]))
        self.register_buffer('clip_code_logvar', torch.zeros([num_train_samples, cfg.POSE2POSE.AUTOENCODER.CODE_DIM]))

        ## regression loss
        self.reg_criterion = nn.L1Loss(reduction='none')
    
    def forward(self, batch, return_loss=True, is_testing=False, interpolation_coeff=None):
        # input
        audio = batch['audio'].cuda()
        num_frames = int(batch['num_frames'][0].item())
        poses_gt_batch = batch['poses'].cuda() if return_loss else None

        # forward
        mel = self.mel_transfm(audio)

        if not return_loss:
            assert self.cfg.DEMO.CODE_PATH is not None
            idx = int((self.cfg.DEMO.MULTIPLE - 1)*interpolation_coeff)
            code = np.load(self.cfg.DEMO.CODE_PATH)['v'][idx] * 10  # 10 is empirically selected

            code = torch.Tensor(code).cuda().unsqueeze(0)
            poses_pred_batch, mu, logvar = self.ae(poses_gt_batch, self.cfg.DATASET.NUM_FRAMES, external_code=code)

            results_dict = {
                'poses_pred_batch': poses_pred_batch,
                'clip_code_mu': mu,
                'clip_code_logvar': logvar,
            }
            return results_dict
        else:
            poses_pred_batch, mu, logvar = self.ae(poses_gt_batch, num_frames, mel)

        losses_dict = {}

        ## autoencoder
        ### regression loss
        reg_loss = self.reg_criterion(poses_pred_batch, poses_gt_batch) * self.cfg.POSE2POSE.LAMBDA_REG
        reg_loss = reg_loss.mean()
        losses_dict['reg_loss'] = reg_loss
        loss = reg_loss.clone()

        ### KL-divergence loss
        kl_loss = 0.5 * (-logvar + mu**2 + torch.exp(logvar) - 1).mean() * self.cfg.POSE2POSE.LAMBDA_KL
        losses_dict['kl_loss'] = kl_loss
        loss = loss + kl_loss
        losses_dict['loss'] = loss

        # collect results
        results_dict = {
            'poses_pred_batch': poses_pred_batch,
            'poses_gt_batch': poses_gt_batch,
            'clip_code_mu': mu,
            'clip_code_logvar': logvar,
        }
        return losses_dict, results_dict

class Pose2Pose(Trainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
    
    def setup_model(self, cfg, state_dict=None):
        if self.is_master_process():
            print(torch.cuda.device_count(), "GPUs are available.")
        print('Setting up models on rank', self.get_rank())
        
        self.model = Pose2PoseModel(cfg, state_dict, self.num_train_samples, self.get_rank()).cuda()
        if self.cfg.SYS.DISTRIBUTED:
            self.model = DDP(self.model, device_ids=[self.get_rank()])
        else:
            self.model = DataParallel(self.model)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
    
    def setup_optimizer(self, checkpoint=None, last_epoch=-1):
        # Optimizer for generator
        ae_parameters = self.model.module.ae.parameters() if isinstance(self.model, (DDP, DataParallel)) \
            else self.model.ae.parameters()

        self.optimizers['optimizer'] = torch.optim.Adam(ae_parameters, lr=self.cfg.TRAIN.LR,
                                                         weight_decay=self.cfg.TRAIN.WD)
        if checkpoint is not None:
            self.optimizers['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
        if self.cfg.TRAIN.LR_SCHEDULER:
            self.schedulers['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizers['optimizer'],
                [self.cfg.TRAIN.NUM_EPOCHS-10, self.cfg.TRAIN.NUM_EPOCHS-2],
                gamma=0.1, last_epoch=last_epoch)

    def train_step(self, batch, t_step, global_step, epoch):
        tag = 'TRAIN'
        dataset = self.train_dataset

        audio = batch['audio']
        speaker_stat = batch['speaker_stat']

        losses_dict, results_dict = self.model(batch)
        results_dict['poses_pred_batch'] = dataset.get_final_results(results_dict['poses_pred_batch'].detach(), speaker_stat)
        results_dict['poses_gt_batch'] = dataset.get_final_results(results_dict['poses_gt_batch'].detach(), speaker_stat)

        clip_indices = batch['clip_index'].cuda()
        self.model.module.clip_code_mu[clip_indices] = results_dict['clip_code_mu'].detach()
        self.model.module.clip_code_logvar[clip_indices] = results_dict['clip_code_logvar'].detach()
        
        losses_dict.update(self.evaluate(results_dict))

        if not self.cfg.SYS.DISTRIBUTED:
            losses_dict = dict(map(lambda x: (x[0], x[1].mean()), losses_dict.items()))
        
        # optimization
        self.optimizers['optimizer'].zero_grad()
        losses_dict['loss'].backward(retain_graph=True)
        self.optimizers['optimizer'].step()

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
                        self.cfg, tag, vid_batch, t_step, epoch, global_step, audio=audio[0].numpy(), 
                            writer=self.tb_writer, base_path=self.base_path)

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

        losses_dict, results_dict = self.model(batch, is_testing=True)
        results_dict['poses_pred_batch'] = dataset.get_final_results(results_dict['poses_pred_batch'].detach(), speaker_stat)
        results_dict['poses_gt_batch'] = dataset.get_final_results(results_dict['poses_gt_batch'].detach(), speaker_stat)

        losses_dict.update(self.evaluate(results_dict, is_testing=True))

        if not self.cfg.SYS.DISTRIBUTED:
            losses_dict = dict(map(lambda x: (x[0], x[1].mean()), losses_dict.items()))

        if self.cfg.SYS.DISTRIBUTED:
            self.reduce_tensor_dict(losses_dict)
        if self.is_master_process():
            if t_step % self.cfg.SYS.LOG_INTERVAL == 0 and self.get_rank() == 0:
                self.logger_writer_step(tag, losses_dict, t_step, epoch)
            
            if t_step % (self.result_saving_interval_test) == 0:
                results_dict = dict(
                    map(lambda x: (x[0], x[1].detach().cpu().numpy()),
                        filter(lambda x: x[1] is not None, 
                            results_dict.items())))
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
        
        batch_losses_dict = dict(map(lambda x: (x[0], x[1] * self.cfg.TEST.BATCH_SIZE), losses_dict.items()))
        batch_results_dict = dict()
        return batch_losses_dict, batch_results_dict
    
    def demo_step(self, batch, t_step, epoch=0, extra_id=None, interpolation_coeff=None):
        tag = 'DEMO'
        dataset = self.test_dataset

        audio = batch['audio']
        speaker_stat = batch['speaker_stat']

        results_dict = self.model(batch, return_loss=False, interpolation_coeff=interpolation_coeff)
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
        
    def evaluate(self, results_dict, is_testing=False):
        anchors = [39, 3, 6]  # head, right hand, left hand
        poses_pred_batch = results_dict['poses_pred_batch']
        poses_gt_batch = results_dict['poses_gt_batch']

        L2_dist = torch.norm(poses_pred_batch-poses_gt_batch, p=2, dim=2)

        # lip sync error
        lip_open_pred = torch.sqrt(
            ((poses_pred_batch[:, :, :, 75] - poses_pred_batch[:, :, :, 71])**2
            ).sum(dim=2))
        lip_open_gt = torch.sqrt(
            ((poses_gt_batch[:, :, :, 75] - poses_gt_batch[:, :, :, 71])**2
            ).sum(dim=2))
        lip_sync_error = torch.abs(lip_open_pred - lip_open_gt)
        # normalized lip error
        lip_open_pred_n = lip_open_pred / (lip_open_gt.max(-1, keepdim=True).values + 1e-4)
        lip_open_gt_n = lip_open_gt / (lip_open_gt.max(-1, keepdim=True).values + 1e-4)
        lip_sync_error_n = torch.abs(lip_open_pred_n - lip_open_gt_n)

        metrics_dict = {
            'L2_dist': L2_dist,
            'lip_sync_error_n': lip_sync_error_n,
        }

        if is_testing and self.cfg.TEST.MULTIPLE > 1:
            multiple = self.cfg.TEST.MULTIPLE

            L2_dist_multiple = L2_dist.reshape((multiple, -1)).mean(1)
            L2_dist_min, L2_dist_max = L2_dist_multiple.min(), L2_dist_multiple.max()

            metrics_dict.update({
                'L2_dist_min': L2_dist_min,
                'L2_dist_max': L2_dist_max,
            })
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

        if self.cfg.POSE2POSE.AUTOENCODER.CODE_DIM is not None:
            code = self.model.module.clip_code_mu.detach().cpu().numpy()

            # kwargs = {'figsize': [6.4, 4.8], 'dpi': 120}
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
    