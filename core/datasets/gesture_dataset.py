import os
import pandas as pd
import numpy as np
import librosa

import torch
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Manager

from ..utils.audio_processing import parse_audio_length, crop_pad_audio
from .speakers_stat import *


class GestureDataset(Dataset):
    def __init__(self, root_dir, speaker, split, cfg, demo_input=None):
        self.cfg = cfg.DATASET
        self.root_dir = os.path.join(root_dir, speaker)
        self.split = split
        assert speaker is not None, 'The speaker is "None"!'
        self.speaker = speaker

        if split == 'train':
            self.clips = self.get_csv_file(self.root_dir)
            self.clips = self.clips[self.clips['dataset'] == 'train']
        elif split == 'val':
            self.clips = self.get_csv_file(self.root_dir)
            self.clips = self.clips[self.clips['dataset'] == 'dev']
        elif split == 'demo':
            if len(demo_input.split()) == 1 and os.path.isdir(demo_input):
                file_list = os.listdir(demo_input)
                np.random.shuffle(file_list)
                file_list = list(filter(lambda x: x.split('.')[-1] == 'wav', file_list[:1000]))[:cfg.DEMO.NUM_SAMPLES]
                self.clips = list(map(lambda x: os.path.join(demo_input, x), file_list))
            else:
                self.clips = demo_input.split()
        else:
            raise NotImplementedError

        if self.cfg.SUBSET is not None:
            self.clips = self.clips[:self.cfg.SUBSET]

        self.root_node = 1  # index in keypoint-122
        self.hand_root_l = 6  # index in keypoint-121 (with the root node removed)
        self.hand_root_r = 3  # index in keypoint-121 (with the root node removed)
        self.head_root = 39  # index in keypoint-121 (with the root node removed)

        if self.cfg.CACHING:
            self.cache_dict = Manager().dict()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        if self.split == 'demo':
            feed = self.clips[idx]
            if feed.split('.')[-1] in ['wav', 'm4a']:
                audio = feed
            else:
                raise NotImplementedError('Audio format %s is not supported.' % feed.split('.')[-1])

            audio, _ = librosa.load(audio, sr=self.cfg.AUDIO_SR)
            if self.cfg.MAX_DEMO_LENGTH is not None:
                max_length = self.cfg.MAX_DEMO_LENGTH * self.cfg.AUDIO_SR
                if len(audio) > max_length:
                    start_point = np.random.randint(0, len(audio)-max_length)
                    audio = audio[start_point:start_point+max_length]
                    
            audio_length, num_frames = parse_audio_length(len(audio), self.cfg.AUDIO_SR, self.cfg.FPS)
            audio = crop_pad_audio(audio, audio_length)

            speaker_stat = self.get_speaker_stat(self.speaker, 121, self.cfg.HIERARCHICAL_POSE)

            sample = {
                'speaker': self.speaker,
                'audio': audio,
                'clip_index': idx,
                'speaker_stat': speaker_stat,
                'num_frames': num_frames,
            }
        else:
            if self.cfg.CACHING:
                if idx in self.cache_dict:
                    return self.cache_dict[idx]

            clip = self.clips.iloc[idx]

            speaker = clip['speaker']
            arr = np.load(os.path.join(self.root_dir, clip['pose_fn']), mmap_mode=None)

            audio = arr['audio']
            audio_length, num_frames = parse_audio_length(self.cfg.AUDIO_LENGTH, self.cfg.AUDIO_SR, self.cfg.FPS)
            audio = crop_pad_audio(audio, audio_length)

            # pose pre-processing
            poses_with_score = torch.Tensor(arr['pose'][:self.cfg.NUM_FRAMES, ...])
            poses_with_score = self.remove_unuesd_kp(poses_with_score)
            relative_poses_with_score = self.absolute_to_relative(poses_with_score)
            if self.cfg.HIERARCHICAL_POSE:
                relative_poses_with_score = self.global_to_parted(relative_poses_with_score)

            relative_poses = relative_poses_with_score[:, :2, :]
            poses_score = relative_poses_with_score[:, 2:, :].repeat(1, 2, 1)
            
            speaker_stat = self.get_speaker_stat(speaker, relative_poses.shape[-1], parted=self.cfg.HIERARCHICAL_POSE)
            normalized_relative_poses = self.normalize_poses(relative_poses, speaker_stat)

            sample = {
                'speaker': speaker,
                'audio': audio,
                'num_frames': num_frames,
                'clip_index': idx,
                'poses': normalized_relative_poses,
                'poses_score': poses_score,
                'speaker_stat': speaker_stat,
                'anchors': {
                    'hand_root_l': self.hand_root_l,
                    'hand_root_r': self.hand_root_r,
                    'head_root': self.head_root},
            }
            if self.cfg.CACHING:
                self.cache_dict[idx] = sample
        return sample

    def get_csv_file(self, root_dir):
        csv_path = os.path.join(root_dir, f'processed_137.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError('No csv file: %s' % csv_path)
        csv_file = pd.read_csv(csv_path)
        return csv_file

    def remove_unuesd_kp(self, poses):
        assert poses.shape[-1] == 137
        # remove lower body
        indices = list(range(0, 8)) + [15, 16] + list(range(25, 137))
        poses = poses[..., :, indices]
        return poses

    def absolute_to_relative(self, poses):
        # relative body poses
        poses[..., :2, :] = poses[..., :2, :] - poses[..., :2, self.root_node, None]

        # remove root node
        indices = [0] + list(range(2, 122))
        poses = poses[..., :, indices]
        return poses

    def parted_to_global(self, poses):
        # global relative head poses
        indices = list(range(9, self.head_root)) + list(range(self.head_root+1, 79))
        poses[..., :2, indices] = poses[..., :2, indices] + poses[..., :2, self.head_root, None]

        # global relative hand poses
        poses[..., :2, 79:100] = poses[..., :2, 79:100] + poses[..., :2, self.hand_root_l, None]
        poses[..., :2, 100:121] = poses[..., :2, 100:121] + poses[..., :2, self.hand_root_r, None]
        return poses
    
    def global_to_parted(self, poses):
        # global relative head poses
        indices = list(range(9, self.head_root)) + list(range(self.head_root+1, 79))
        poses[..., :2, indices] = poses[..., :2, indices] - poses[..., :2, self.head_root, None]

        # global relative hand poses
        poses[..., :2, 79:100] = poses[..., :2, 79:100] - poses[..., :2, self.hand_root_l, None]
        poses[..., :2, 100:121] = poses[..., :2, 100:121] - poses[..., :2, self.hand_root_r, None]
        return poses

    def get_speaker_stat(self, speaker, num_kp, parted):
        if parted:
            return eval(f'SPEAKERS_STAT_{num_kp}_parted')[speaker]
        else:
            return eval(f'SPEAKERS_STAT_{num_kp}')[speaker]

    def normalize_poses(self, kp, speaker_stat):
        if isinstance(speaker_stat['mean'], np.ndarray):
            mean = torch.Tensor(speaker_stat['mean'].astype(np.float)).to(kp.device)
            std = torch.Tensor(speaker_stat['std'].astype(np.float)).to(kp.device)
        elif isinstance(speaker_stat['mean'], torch.Tensor):
            mean = speaker_stat['mean'].to(kp.device)
            std = speaker_stat['std'].to(kp.device)
        
        if mean.dim() == 1:
            mean = mean.reshape(1, 2, self.cfg.NUM_LANDMARKS)
            std = std.reshape(1, 2, self.cfg.NUM_LANDMARKS)
        elif mean.dim() == 2:
            mean = mean.reshape(kp.shape[0], 1, 2, self.cfg.NUM_LANDMARKS)
            std = std.reshape(kp.shape[0], 1, 2, self.cfg.NUM_LANDMARKS)
        else:
            raise NotImplementedError
        
        kp = (kp - mean) / std
        return kp
    
    def denormalize_poses(self, kp, speaker_stat):
        if isinstance(speaker_stat['mean'], np.ndarray):
            mean = torch.Tensor(speaker_stat['mean'].astype(np.float)).to(kp.device)
            std = torch.Tensor(speaker_stat['std'].astype(np.float)).to(kp.device)
        elif isinstance(speaker_stat['mean'], torch.Tensor):
            mean = speaker_stat['mean'].to(kp.device)
            std = speaker_stat['std'].to(kp.device)
        
        if mean.dim() == 1:
            mean = mean.reshape(1, 2, self.cfg.NUM_LANDMARKS)
            std = std.reshape(1, 2, self.cfg.NUM_LANDMARKS)
        elif mean.dim() == 2:
            mean = mean.reshape(kp.shape[0], 1, 2, self.cfg.NUM_LANDMARKS)
            std = std.reshape(kp.shape[0], 1, 2, self.cfg.NUM_LANDMARKS)
        else:
            raise NotImplementedError

        kp = kp * std + mean
        return kp
    
    def get_final_results(self, poses, speaker_stat):
        poses = self.denormalize_poses(poses, speaker_stat)
        if self.cfg.HIERARCHICAL_POSE:
            poses = self.parted_to_global(poses)

        scale_factor = speaker_stat['scale_factor'].to(poses.device).reshape(speaker_stat['scale_factor'].shape[0], 1, 1, -1)
        poses = poses * scale_factor
        return poses
    
    def transform_normalized_parted2global(self, poses, speaker):
        ''' transform a non-hierarchical prediction into a hierarchical one
        
        This is a temporal function to prepare input for FGD.
        Here we assume that the speakers in a batch are the same.
        (WARNING: will be deprecated in the future!)
        '''
        speaker_stat_global = self.get_speaker_stat(speaker[0], poses.shape[-1], False)
        speaker_stat_parted = self.get_speaker_stat(speaker[0], poses.shape[-1], True)

        poses = self.denormalize_poses(poses, speaker_stat_parted)
        poses = self.parted_to_global(poses)

        poses = self.normalize_poses(poses, speaker_stat_global)
        return poses
    

if __name__ == "__main__":
    from configs.default import get_cfg_defaults
    from core.utils.keypoint_visualization import vis_relative_pose_clip
    import cv2

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/voice2pose_sdt_bp.yaml')
    cfg.freeze()
    print(cfg)

    dataset = GestureDataset(cfg.DATASET.ROOT_DIR, cfg.DATASET.SPEAKER, 'train', cfg)
    dataset = GestureDataset(cfg.DATASET.ROOT_DIR, cfg.DATASET.SPEAKER, 'val', cfg)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=64)

    k = None
    for sample in dataloader:
        idx = sample['clip_index']
        audio = sample['audio']
        poses_gt = sample['poses']
        audio_path = sample['audio_path']
        speaker_stat = sample['speaker_stat']
        normalized_relative_poses = sample['poses']

        mean = sample['speaker_stat']['mean'][0].reshape(1, 1, 2, -1)
        std = sample['speaker_stat']['std'][0].reshape(1, 1, 2, -1)
        relative_poses = normalized_relative_poses * std + mean
        if cfg.DATASET.HIERARCHICAL_POSE:
            transform_func = dataset.parted_to_global
        relative_poses = transform_func(relative_poses)

        img_list = vis_relative_pose_clip(relative_poses[0], (1080, 1920))
        for img in img_list:
            img = cv2.resize(img, (1280, 720))
            cv2.imshow('0', img)
            k = cv2.waitKey(1)
        
        if k == ord('q'):
            break


