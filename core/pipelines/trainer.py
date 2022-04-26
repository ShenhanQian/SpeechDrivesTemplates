import os
from datetime import datetime
import logging
import time
from abc import abstractmethod
import matplotlib as mpl
import numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from core.datasets import get_dataset
from core.networks import get_model
from core.utils.video_processing import VideoWriter


class Trainer(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = None
        self.optimizers = {}
        self.schedulers = {}
        self.train_dataloader = None
        self.test_dataloader = None
        torch.cuda.set_device(self.get_rank())
    
    def get_rank(self):
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        else:
            return 0
    
    def get_world_size(self):
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        else:
            return 1
    
    def is_master_process(self):
        if not torch.distributed.is_initialized():
            return True
        else:
            return not self.get_rank()
    
    def setup_logger(self, base_path, exp_name):
        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.INFO)

        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-0.5s] %(message)s")

        log_path = "{0}/{1}.log".format(base_path, exp_name)
        fileHandler = logging.FileHandler(log_path)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

        logging.info('log path: %s' % log_path)
    
    def setup_dataset(self, cfg, split, demo_input=None):
        if self.is_master_process():
            print('Setting up dataset...')

        if split == 'train':
            self.train_dataset = get_dataset(cfg.DATASET.NAME)(self.cfg.DATASET.ROOT_DIR, self.cfg.DATASET.SPEAKER, 'train', self.cfg)
            if self.cfg.SYS.DISTRIBUTED:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            else:
                self.train_sampler = None
            self.train_dataloader = DataLoader(self.train_dataset,
                batch_size=self.cfg.TRAIN.BATCH_SIZE // self.get_world_size(),
                shuffle=(self.train_sampler is None),
                num_workers=self.cfg.SYS.NUM_WORKERS // self.get_world_size(),
                sampler=self.train_sampler, drop_last=True)
            self.num_train_samples = len(self.train_dataset)
            self.num_train_batches = len(self.train_dataloader)
            self.result_saving_interval_train = self.num_train_batches // self.cfg.TRAIN.NUM_RESULT_SAMPLE \
                if self.num_train_batches // self.cfg.TRAIN.NUM_RESULT_SAMPLE > 0 else 1
                
            if self.is_master_process():
                print('num_train_samples: %d' % self.num_train_samples)

            if self.cfg.TRAIN.VALIDATE:
                self.test_dataset = get_dataset(cfg.DATASET.NAME)(self.cfg.DATASET.ROOT_DIR, self.cfg.DATASET.SPEAKER, 'val', self.cfg)
                if self.cfg.SYS.DISTRIBUTED:
                    self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False)
                else:
                    self.val_sampler = None
                self.test_dataloader = DataLoader(self.test_dataset,
                    batch_size=self.cfg.TEST.BATCH_SIZE // self.get_world_size(),
                    shuffle=False,
                    num_workers=self.cfg.SYS.NUM_WORKERS // self.get_world_size(),
                    sampler=self.val_sampler)
                self.num_test_samples = len(self.test_dataset)
                self.num_test_batches = len(self.test_dataloader)
                self.result_saving_interval_test = self.num_test_batches // self.cfg.TEST.NUM_RESULT_SAMPLE \
                    if self.num_test_batches // self.cfg.TEST.NUM_RESULT_SAMPLE > 0 else 1
                if self.is_master_process():
                    print('num_val_samples: %d' % self.num_test_samples)

        elif split == 'test':
            self.num_train_samples = None

            self.test_dataset = get_dataset(cfg.DATASET.NAME)(self.cfg.DATASET.ROOT_DIR, self.cfg.DATASET.SPEAKER, 'val', self.cfg)
            if self.cfg.SYS.DISTRIBUTED:
                self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False)
            else:
                self.test_sampler = None
            self.test_dataloader = DataLoader(self.test_dataset,
                batch_size=self.cfg.TEST.BATCH_SIZE // self.get_world_size(),
                shuffle=False,
                num_workers=self.cfg.SYS.NUM_WORKERS // self.get_world_size(),
                sampler=self.test_sampler, drop_last=False)
            self.num_test_samples = len(self.test_dataset)
            self.num_test_batches = len(self.test_dataloader)
            self.result_saving_interval_test = self.num_test_batches // self.cfg.TEST.NUM_RESULT_SAMPLE \
                if self.num_test_batches // self.cfg.TEST.NUM_RESULT_SAMPLE > 0 else 1
            if self.is_master_process():
                print('num_test_samples: %d' % self.num_test_samples)
        
        elif split == 'demo':
            self.num_train_samples = None

            self.test_dataset = get_dataset(cfg.DATASET.NAME)(self.cfg.DATASET.ROOT_DIR, self.cfg.DATASET.SPEAKER, 'demo', self.cfg, demo_input=demo_input)
            if self.cfg.SYS.DISTRIBUTED:
                self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False)
            else:
                self.test_sampler = None
            self.test_dataloader = DataLoader(self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.cfg.SYS.NUM_WORKERS // self.get_world_size(),
                sampler=self.test_sampler)
            self.num_test_samples = len(self.test_dataset)
            self.num_test_batches = len(self.test_dataloader)
            self.result_saving_interval_test = self.num_test_batches // self.cfg.TEST.NUM_RESULT_SAMPLE \
                if self.num_test_batches // self.cfg.TEST.NUM_RESULT_SAMPLE > 0 else 1
            if self.is_master_process():
                print('num_test_samples: %d' % self.num_test_samples)
        else:
            raise Exception('Unknown data split.')

    @abstractmethod
    def setup_model(self, cfg, state_dict=None):
        pass

    def setup_optimizer(self, checkpoint=None, last_epoch=-1):
        learning_rate = self.cfg.TRAIN.LR
        self.optimizers['optimizer'] = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if checkpoint is not None:
            self.optimizers['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])

        if self.cfg.TRAIN.LR_SCHEDULER:
            self.schedulers['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizers['optimizer'], [90, 98], gamma=0.1, last_epoch=last_epoch)

    def setup_experiment(self, is_training, exp_tag, resume_from=None, checkpoint=None, demo_input=None):
        map_location = {'cuda:0' : 'cuda:%d' % self.get_rank()}
        if self.is_master_process():
            print('Setting up base directory...')
        dt = str(datetime.now()).replace('.', '-').replace(':', '-').replace(' ', '_')
        exp_tag = '_'.join([dt, exp_tag])

        if is_training:
            self.setup_dataset(self.cfg, 'train')

            if resume_from is not None:
                assert resume_from.split('.')[-1]=='pth', 'file type not supported: %s' % resume_from
                assert os.path.exists(resume_from), 'file not exists: %s' % resume_from
                if self.is_master_process():
                    print('Resuming from checkpoint: %s' % resume_from)
                checkpoint = torch.load(resume_from, map_location=map_location)
                
                epoch = checkpoint['epoch']
                global_step = checkpoint['step']
                base_path = os.path.split(resume_from)[0]
                
                self.setup_model(self.cfg, state_dict=checkpoint['model_state_dict'])
                self.setup_optimizer(checkpoint=checkpoint, last_epoch=epoch)
            else:
                epoch = 0
                global_step = 0
                base_path = os.path.join(self.cfg.SYS.OUTPUT_DIR, exp_tag)  
                if self.is_master_process():
                    os.makedirs(base_path)

                if self.cfg.TRAIN.PRETRAIN_FROM is not None:
                    pretrain_from = self.cfg.TRAIN.PRETRAIN_FROM
                    assert pretrain_from.split('.')[-1]=='pth', 'file type not supported: %s' % pretrain_from
                    assert os.path.exists(pretrain_from), 'file not exists: %s' % pretrain_from
                    if self.is_master_process():
                        print('Loading from pretrained model: %s' % pretrain_from)
                    checkpoint = torch.load(pretrain_from, map_location=map_location)
                    
                    self.setup_model(self.cfg, state_dict=checkpoint['model_state_dict'])
                else:
                    self.setup_model(self.cfg)
                self.setup_optimizer()
            return base_path, epoch, global_step
        else:
            if demo_input is None:
                self.setup_dataset(self.cfg, 'test')
            else:
                self.setup_dataset(self.cfg, 'demo', demo_input=demo_input)

            base_path = os.path.join(self.cfg.SYS.OUTPUT_DIR, exp_tag)  
            if self.is_master_process():
                os.makedirs(base_path)

            if checkpoint is not None:
                print('Loading from checkpoint: %s' % checkpoint)

                assert checkpoint.split('.')[-1]=='pth', 'file type not supported: %s' % checkpoint
                assert os.path.exists(checkpoint), 'file not exists: %s' % checkpoint
                checkpoint = torch.load(checkpoint)
                self.setup_model(self.cfg, state_dict=checkpoint['model_state_dict'])
            else:
                raise Exception('Checkpoint file is not provided.')
        return base_path
    
    @abstractmethod
    def draw_figure_epoch(self):
        fig_dict = {}
        mpl.use('Agg')
        mpl.rcParams['agg.path.chunksize'] = 10000
        return fig_dict
    
    @abstractmethod
    def evaluate_epoch(self, results_dict):
        tic = time.time()
        metrics_dict = {}
        
        toc = time.time() - tic
        logging.info('Compelte epoch evaluation in %.2f min' % (toc/60))
        return metrics_dict
    
    def logger_writer_step(self, tag, losses, step, epoch=None, global_step=None):
        step_toc = (time.time() - self.step_tic) / self.cfg.SYS.LOG_INTERVAL
        self.step_tic = time.time()

        if tag == 'TRAIN':
            msg = '[%s] epoch: %d/%d  step: %d/%d  global_step: %d  time: %.3f  '% (
                tag, epoch, self.cfg.TRAIN.NUM_EPOCHS, step, self.num_train_batches, global_step, step_toc)
            # learning rate
            for k, v in self.optimizers.items():
                lr_values = list(map(lambda x: x['lr'], v.param_groups))
                for i, lr in enumerate(lr_values):
                    if i == 0:
                        msg += 'lr_%s: %.1e  ' % (k, lr)
                        self.tb_writer.add_scalar('train/lr_%s' % k, lr, global_step)
                    else:
                        msg += 'lr_%s_%d: %.1e  ' % (k, i, lr)
                        self.tb_writer.add_scalar('train/lr_%s_%d' % (k, i), lr, global_step)
            # losses
            for k, v in losses.items():
                loss = v.detach().cpu().numpy()
                msg += '%s: %.5f  ' % (k, v)
                self.tb_writer.add_scalar('train/%s' % (k), loss, global_step)
            
        elif tag == 'VAL' or tag == 'TEST':  # testing or validation
            msg = '[%s] epoch: %d/%d  step: %d/%d  time: %.3f  ' % (
                    tag, epoch, self.cfg.TRAIN.NUM_EPOCHS, step, self.num_test_batches, step_toc)
            # losses
            msg += ''.join(['%s: %.5f  ' % (k, v) for k, v in losses.items()])

        else:
            raise Exception('Unknown tag:', tag)

        logging.info(msg)
    
    def logger_writer_epoch(self, tag, epoch_toc, losses=None, figures=None, epoch=0, ETA=None):
        if tag == 'TRAIN':
            msg = '[TRAIN] epoch_time: %.2f hours  ETA: %.2f hours' % (epoch_toc, ETA)
            self.tb_writer.add_scalar('train/epoch_time', epoch_toc, global_step=epoch)
            self.tb_writer.add_scalar('train/ETA', ETA, global_step=epoch)
            # figures
            for k, v in figures.items():
                self.tb_writer.add_figure('%s/%s' % (tag.lower(), k), v, global_step=epoch)
            
        elif tag == 'VAL' or tag == 'TEST':  # testing or validation
            epoch_counter = 'epoch: %d/%d  ' %(epoch, self.cfg.TRAIN.NUM_EPOCHS, ) \
                if tag == 'VAL' else ''
            msg = '[%s] %sval_time: %.1f min  num_samples: %d  ' % (
                tag, epoch_counter, epoch_toc, self.num_test_samples)
            # losses
            for k, v in losses.items():
                loss = v.detach().cpu().numpy()
                msg += '%s: %.5f  ' % (k, v)
                self.tb_writer.add_scalar('%s/%s' % (tag.lower(), k), loss, global_step=epoch)
        
        elif tag == 'DEMO':  # demo
            msg = '[%s] time: %.1f min  num_samples: %d  ' % (
                tag, epoch_toc, self.num_test_samples)

        else:
            raise Exception('Unknown tag:', tag)

        logging.info(msg)
    
    def save_checkpoint(self, epoch, global_step):
        checkpoint_dir = os.path.join(self.base_path, 'checkpoints') 
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, 
            'checkpoint_epoch-%d_step-%d.pth' % (epoch, global_step))
        logging.info('Saving checkpoint to: %s' % checkpoint_path)
        
        ckpt_dict = {
            'epoch': epoch,
            'step': global_step,
            'model_state_dict': self.model.state_dict(),
            }
        for k, v in self.optimizers.items():
            ckpt_dict['%s_state_dict' % k] = v.state_dict()
        
        torch.save(ckpt_dict, checkpoint_path)
    
    def reduce_tensor_dict(self, tensor_dict):
        for k, v in tensor_dict.items():
            torch.distributed.reduce(v, 0)
            if self.is_master_process():
                tensor_dict[k] = v / self.get_world_size()
    
    def concat_tensor_dict(self, input_dict, collection_dict):
        for k, v in input_dict.items():
            assert isinstance(v, np.ndarray) or isinstance(v, torch.Tensor)
            if k not in collection_dict:
                collection_dict[k] = v
            else:
                if isinstance(v, np.ndarray):
                    collection_dict[k] = np.concatenate([collection_dict[k], v], axis=0)
                elif isinstance(v, torch.Tensor):
                    collection_dict[k] = torch.cat([collection_dict[k], v], dim=0)
                else:
                    raise NotImplementedError
        return collection_dict
    
    def mutiply_batch(self, batch, multiple):
        if isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.mutiply_batch(v, multiple)
            return batch
        elif isinstance(batch, list):
            return batch * multiple
        elif isinstance(batch, torch.Tensor):
            return batch.unsqueeze(0).repeat_interleave(multiple, dim=0).reshape(multiple*batch.shape[0], *batch.shape[1:])
        else:
            raise NotImplementedError
    
    @abstractmethod
    def train_step(self, batch, t_step, global_step, epoch):
        pass
    
    @abstractmethod
    def test_step(self, batch, t_step, epoch=0):
        pass

    @abstractmethod
    def demo_step(self, batch, t_step, epoch=0, extra_id=None, interpolation_coeff=None):
        pass
    
    def train(self, cfg, exp_tag, resume_from):
        self.base_path, epoch, global_step = self.setup_experiment(True, exp_tag, resume_from=resume_from)

        if self.is_master_process():
            print('Setting up logger and summary writer...')
            self.setup_logger(self.base_path, exp_tag)
            self.tb_writer = SummaryWriter(log_dir=self.base_path)
            self.video_writer = VideoWriter(self.cfg)
            logging.info('\n====== Configurations ======\n' + str(cfg) + '\n============\n')
            logging.info('Training begins!')
        while epoch < cfg.TRAIN.NUM_EPOCHS:
            epoch += 1
            epoch_tic = time.time()
            epoch_toc_list = []
            self.step_tic = time.time()
            self.model.train()
            if self.cfg.SYS.DISTRIBUTED:
                self.train_sampler.set_epoch(epoch)
            for t_step, batch in enumerate(self.train_dataloader):
                global_step += 1
                self.train_step(batch, t_step+1, global_step, epoch)

            if epoch % cfg.TRAIN.CHECKPOINT_INTERVAL == 0:
                if self.get_rank() == 0:
                    self.save_checkpoint(epoch, global_step)

                if cfg.TRAIN.VALIDATE:
                    self.validate(self.test_dataloader, epoch)
            
            if self.cfg.TRAIN.LR_SCHEDULER:
                for _, v in self.schedulers.items():
                    v.step()
            epoch_toc = (time.time() - epoch_tic) / 3600
            epoch_toc_list.append(epoch_toc)
            epoch_toc_mean = sum(epoch_toc_list) / len(epoch_toc_list) if len(epoch_toc_list) < 10 else sum(epoch_toc_list[-10:]) / 10
            ETA = (self.cfg.TRAIN.NUM_EPOCHS - epoch) * epoch_toc_mean
            if self.is_master_process():
                fig_dict = self.draw_figure_epoch()
                self.logger_writer_epoch('TRAIN', epoch_toc, epoch=epoch, ETA=ETA, figures=fig_dict)

    def validate(self, test_dataloader, epoch):
        if self.is_master_process():
            logging.info('Validation begins!')
        epoch_tic = time.time()
      
        self.model.eval() 
        with torch.no_grad():
            losses_sum_dict = {}
            epoch_results_dict = {}
            self.step_tic = time.time()
            for v_step, batch in enumerate(test_dataloader):
                batch_losses_dict, batch_results_dict = self.test_step(batch, v_step+1, epoch=epoch)
                for k, v in batch_losses_dict.items():
                    losses_sum_dict[k] = losses_sum_dict[k]+v if k in losses_sum_dict else v
                epoch_results_dict = self.concat_tensor_dict(batch_results_dict, epoch_results_dict)
            losses_epoch_dict = dict(map(lambda x: (x[0], (x[1] / self.num_test_samples)), losses_sum_dict.items()))
            losses_epoch_dict.update(self.evaluate_epoch(epoch_results_dict))

        epoch_toc = (time.time() - epoch_tic) / 60
        if self.is_master_process():
            self.logger_writer_epoch('VAL', epoch_toc, epoch=epoch, losses=losses_epoch_dict)

    def test(self, cfg, exp_tag, checkpoint):
        if self.is_master_process():
            print('Setting up logger and summary writer...')
        self.base_path = self.setup_experiment(False, exp_tag, checkpoint=checkpoint)

        if self.is_master_process():
            self.setup_logger(self.base_path, exp_tag)
            self.tb_writer = SummaryWriter(log_dir=self.base_path)
            self.video_writer = VideoWriter(self.cfg)
            logging.info('\n====== Configurations ======\n' + str(cfg) + '\n============\n')
            logging.info('Testing begins!\n')
        epoch_tic = time.time()

        self.model.eval()
        with torch.no_grad():
            losses_sum_dict = {}
            epoch_results_dict = {}
            self.step_tic = time.time()
            for t_step, batch in enumerate(self.test_dataloader):
                batch_losses_dict, batch_results_dict = self.test_step(batch, t_step+1, epoch=0)
                for k, v in batch_losses_dict.items():
                    losses_sum_dict[k] = losses_sum_dict[k]+v if k in losses_sum_dict else v
                epoch_results_dict = self.concat_tensor_dict(batch_results_dict, epoch_results_dict)
            losses_epoch_dict = dict(map(lambda x: (x[0], (x[1] / self.num_test_samples)), losses_sum_dict.items()))
            losses_epoch_dict.update(self.evaluate_epoch(epoch_results_dict))

        epoch_toc = (time.time() - epoch_tic) / 60
        if self.is_master_process():
            self.logger_writer_epoch('TEST', epoch_toc, losses=losses_epoch_dict)
    
    def demo(self, cfg, exp_tag, checkpoint, demo_input):
        if self.is_master_process():
            print('Setting up logger and summary writer...')
        self.base_path = self.setup_experiment(False, exp_tag, checkpoint=checkpoint, demo_input=demo_input)

        if self.is_master_process():
            self.setup_logger(self.base_path, exp_tag)
            self.tb_writer = SummaryWriter(log_dir=self.base_path)
            self.video_writer = VideoWriter(self.cfg)
            logging.info('\n====== Configurations ======\n' + str(cfg) + '\n============\n')
            logging.info('Demo begins!\n')
        epoch_tic = time.time()

        self.model.eval()
        with torch.no_grad():
            self.step_tic = time.time()
            for t_step, batch in enumerate(self.test_dataloader):
                if self.cfg.DEMO.MULTIPLE > 1:
                    for i in range(self.cfg.DEMO.MULTIPLE):
                        self.demo_step(batch, t_step+1, epoch=0, extra_id=i, interpolation_coeff=i/(self.cfg.DEMO.MULTIPLE-1))
                else:
                    self.demo_step(batch, t_step+1, epoch=0)

        epoch_toc = (time.time() - epoch_tic) / 60
        if self.is_master_process():
            self.logger_writer_epoch('DEMO', epoch_toc)
