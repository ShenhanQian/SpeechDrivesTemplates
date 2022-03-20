import argparse
import shutil
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool, RLock, freeze_support


def pose137_to_pose121(x):
    return np.concatenate([x[:, 0:1], x[:, 2:8],  # upper_body
                           x[:, 15:17],   # eyes
                           x[:, 25:]], axis=-1)   # face, hand_l and hand_r


def check_is_pose_outlier(f_path):
    """User can specify their own outlier pose condition here"""

    pose_np = np.load(f_path)  # shape (3, 137)
    pose_np = pose137_to_pose121(pose_np)
    for i in range(121):
        if (pose_np[:2, i] <= 3).all():
            return True
    return False


def clean_pose_per_video(video_path):
    # print(f"cleaning pose for video: {video_path}")
    ls_pose_f = sorted(os.listdir(video_path))
    for pose_fn in tqdm(ls_pose_f, desc=f"cleaning pose for video: {video_path}"):
        pose_fp = os.path.join(video_path, pose_fn)
        if check_is_pose_outlier(pose_fp):
            os.remove(pose_fp)


def clean_pose_per_video_multiprocess_single(pose_fp):
    if check_is_pose_outlier(pose_fp):
        os.remove(pose_fp)


def clean_pose_per_video_multiprocess(video_path, pool):
    print(f"cleaning pose for video: {video_path}")
    ls_pose_fn = sorted(os.listdir(video_path))
    ls_pose_fp = [os.path.join(video_path, pose_fn) for pose_fn in ls_pose_fn]
    pool.map(clean_pose_per_video_multiprocess_single, ls_pose_fp)


parser = argparse.ArgumentParser(description='remove outliers')
parser.add_argument('-b', '--base_dataset_path', type=str, default=None, help="dataset root path", required=True)
parser.add_argument('-s', '--speaker', type=str, default='Default Speaker Name', required=True)

parser.add_argument('-np', '--num_processes', type=int, default=1)
parser.add_argument("-d", "--debug", help="debug mode", action="store_true")
args = parser.parse_args()


DATASET_PATH = os.path.join(args.base_dataset_path, args.speaker)
DIR_RAW_POSE = os.path.join(DATASET_PATH, "tmp", "raw_pose_2d")
DIR_CLEANED_POSE = os.path.join(DATASET_PATH, "tmp", "cleaned_pose_2d")


if __name__ == "__main__":
    if not os.path.exists(DIR_CLEANED_POSE):
        print("Copying dir_raw_pose to dir_cleaned_pose...")
        # shutil.copytree(DIR_RAW_POSE, DIR_CLEANED_POSE)
        command = f"cp -r {DIR_RAW_POSE} {DIR_CLEANED_POSE}"
        print(f'command: {command}')
        os.system(command)
    else:
        print("cleaned_pose_2d dir already exists")

    ls_vid = sorted(os.listdir(DIR_CLEANED_POSE))
    if args.num_processes > 1:
        freeze_support()
        p = Pool(args.num_processes, initializer=tqdm.set_lock, initargs=(RLock(),))
        # p = Pool(args.num_processes)
        for vid_nm in tqdm(ls_vid):
            dir_vid = os.path.join(DIR_CLEANED_POSE, vid_nm)
            clean_pose_per_video_multiprocess(dir_vid, p)
    else:
        for vid_nm in tqdm(ls_vid):
            dir_vid = os.path.join(DIR_CLEANED_POSE, vid_nm)
            clean_pose_per_video(dir_vid)
