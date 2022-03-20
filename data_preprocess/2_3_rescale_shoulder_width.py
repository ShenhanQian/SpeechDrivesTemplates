import argparse
import numpy as np
import os
from multiprocessing import Pool, RLock
import tqdm
import shutil


parser = argparse.ArgumentParser(description='remove outliers')
parser.add_argument('-b', '--base_dataset_path', default=None, help="dataset root path", required=True)
parser.add_argument('-s', '--speaker', default='Default Speaker Name', required=True)

parser.add_argument('-np', '--num_processes', type=int, default=1)
parser.add_argument("-d", "--debug", help="debug mode", action="store_true")
args = parser.parse_args()


DATASET_PATH = os.path.join(args.base_dataset_path, args.speaker)
DIR_CLEANED_POSE = os.path.join(DATASET_PATH, "tmp", "cleaned_pose_2d")
DIR_RESCALED_POSE = os.path.join(DATASET_PATH, "tmp", "rescaled_pose_2d")


def cal_shoulder_distance(np_pose):
    """np_pose.shape: [3, 137]"""
    return np.sqrt(np.sum((np_pose[:2, 2] - np_pose[:2, 5]) ** 2))


def cal_mean_shoulder_distance_single_process(lst_in):
    ls_pose_fn = lst_in[0]

    np_avg = 0
    num = 0
    for pose_fn in ls_pose_fn:
        try:
            np_pose = np.load(pose_fn)
        except:
            print(f"ERROR: File CANNOT open: {pose_fn}")
            exit(1)
        weight = num / (num + 1)
        np_avg = np_avg * weight + (1 - weight) * cal_shoulder_distance(np_pose)
        num += 1

    return np_avg


def cal_mean_shoulder_distance(vid_dir):
    ls_pose_all = sorted(os.listdir(vid_dir))
    ls_pose_all = [os.path.join(vid_dir, i) for i in ls_pose_all]
    num_thread = args.num_processes
    stride = len(ls_pose_all) // num_thread

    lst_pose = [(ls_pose_all[i * stride: (i + 1) * stride], i) for i in range(num_thread)]

    p = Pool(num_thread, initializer=tqdm.tqdm.set_lock, initargs=(RLock(),))
    ans = p.map(cal_mean_shoulder_distance_single_process, lst_pose)
    p.close()
    p.join()
    ans = np.array(ans)
    mean_shoulder_distance = np.average(ans, axis=0)

    return mean_shoulder_distance


def cal_speaker_scalar_2oliver(vid_dir):
    print(f"Calculating speaker scale factor for :{vid_dir}")

    oliver_scalar = 1.0
    oliver_shoulder_dist = 331.0850066245443

    speaker_shoulder_distance = cal_mean_shoulder_distance(vid_dir)
    speaker_scalar = oliver_shoulder_dist * oliver_scalar / speaker_shoulder_distance
    return speaker_scalar


def override_pose_file_with_scalar(tuple_in):
    fn_npy, scalar = tuple_in
    pose_np = np.load(fn_npy)
    pose_np[:2, :] = pose_np[:2, :] * scalar
    np.save(fn_npy, pose_np)


def rescale_shoulder_width_per_video(vid_dir):
    scalar = cal_speaker_scalar_2oliver(vid_dir)
    ls_npy = sorted(os.listdir(vid_dir))
    ls_npy = [os.path.join(vid_dir, fn_npy) for fn_npy in ls_npy]

    print(f"Overriding pose files with rescaled poses for : {vid_dir} with scalar :{scalar}")
    ls_param = [(fn_npy, scalar) for fn_npy in ls_npy]

    num_thread = args.num_processes
    if num_thread == 1:
        for fn_npy in tqdm.tqdm(ls_npy):
            pose_np = np.load(fn_npy)
            pose_np = pose_np * scalar
            np.save(fn_npy, pose_np)
    else:
        p = Pool(num_thread)
        p.map(override_pose_file_with_scalar, ls_param)
        p.close()
        p.join()


if __name__ == "__main__":
    if not os.path.exists(DIR_RESCALED_POSE):
        print(f"Copying dir_rescaled_pose...")
        # shutil.copytree(DIR_CLEANED_POSE, DIR_RESCALED_POSE)
        command = f"cp -r {DIR_CLEANED_POSE} {DIR_RESCALED_POSE}"
        print(f'command: {command}')
        os.system(command)
    else:
        print(f"dir_rescaled_pose already exists.")

    ls_vid_nm = sorted(os.listdir(DIR_RESCALED_POSE))
    ls_vid_dir = [os.path.join(DIR_RESCALED_POSE, vid_nm) for vid_nm in ls_vid_nm]
    for vid_dir in tqdm.tqdm(ls_vid_dir, desc="Main Process"):
        rescale_shoulder_width_per_video(vid_dir)
