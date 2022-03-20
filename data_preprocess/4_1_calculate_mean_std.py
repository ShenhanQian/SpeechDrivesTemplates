import argparse
import json
import os
from multiprocessing import Pool, RLock, freeze_support

import numpy as np
import pandas as pd
import tqdm


def cal_mean_global(tuple_in):
    df_pose, idx = tuple_in
    num = np.zeros((64, 137))

    np_avg = np.zeros((64, 2, 137))
    for i_pose_fn in tqdm.tqdm(df_pose, desc="m # {}: ".format(idx), position=idx):
        pose_np = np.load(i_pose_fn)["pose"]
        for i in range(64):
            save_pose_root = pose_np[i, :2, 1]
            pose_np[i, :2, 0:1] -= pose_np[i, :2, 1, None]
            pose_np[i, :2, 2:] -= pose_np[i, :2, 1, None]
            for i_kpt in range(137):

                # in case the pose_np is not detected, the pose_np value would be too small, which would effect the
                # -- calculation of mean and std
                if abs(pose_np[i, 0, i_kpt] + save_pose_root) < 5 and \
                        abs(pose_np[i, 1, i_kpt] + save_pose_root) < 5:
                    continue
                weight = num[i, i_kpt] / (num[i, i_kpt] + 1)
                np_avg[i, :, i_kpt] = np_avg[i, :, i_kpt] * weight + (1 - weight) * (pose_np[i, :2, i_kpt])
                num[i, i_kpt] += 1
    return np_avg


def cal_std_global(tuple_in):
    np_avg, df_pose, idx = tuple_in
    num = np.zeros((64, 137))
    np_var = np.zeros((64, 2, 137))
    for i_pose_fn in tqdm.tqdm(df_pose, desc="v # {}: ".format(idx), position=idx):
        pose_np = np.load(i_pose_fn)["pose"]
        for i in range(64):
            save_pose_root = pose_np[i, :2, 1]
            pose_np[i, :2, 0:1] -= pose_np[i, :2, 1, None]
            pose_np[i, :2, 2:] -= pose_np[i, :2, 1, None]
            for i_kpt in range(137):

                # in case the pose_np is not detected, the pose_np value would be too small, which would effect the
                # -- calculation of mean and std
                if abs(pose_np[i, 0, i_kpt] + save_pose_root[0]) < 5 and \
                        abs(pose_np[i, 1, i_kpt] + save_pose_root[1]) < 5:
                    continue
                weight = num[i, i_kpt] / (num[i, i_kpt] + 1)
                np_var[i, :, i_kpt] = np_var[i, :, i_kpt] * weight + (1 - weight) * (
                        pose_np[i, :2, i_kpt] - np_avg[i, :2, i_kpt]) ** 2
                num[i, i_kpt] += 1
    return np.sqrt(np_var)


def pose_np_deduct_root(pose_np, i):
    # make all nodes are rooted on the root node (node #1, end of the neck)
    save_pose_root = pose_np[i, :2, 1]
    pose_np[i, :2, 0:1] -= pose_np[i, :2, 1, None]
    pose_np[i, :2, 2:] -= pose_np[i, :2, 1, None]

    # nodes of hands are rooted on the wrists
    pose_np[i, :2, 95:116] -= pose_np[i, :2, 7:8]
    pose_np[i, :2, 116:137] -= pose_np[i, :2, 4:5]

    # nodes of face are rooted on nose (node #(30+25), except node #(30+25) itself
    pose_np[i, :2, 25:55] -= pose_np[i, :2, 55:56]
    pose_np[i, :2, 56:95] -= pose_np[i, :2, 55:56]
    return pose_np, save_pose_root


def cal_mean_parted(tuple_in):
    df_pose, idx = tuple_in
    num = np.zeros((64, 137))

    np_avg = np.zeros((64, 2, 137))
    for i_pose_fn in tqdm.tqdm(df_pose, desc="m # {}: ".format(idx), position=idx):
        pose_np = np.load(i_pose_fn)["pose"]
        for i in range(64):
            pose_np, save_pose_root = pose_np_deduct_root(pose_np, i)
            for i_kpt in range(137):

                # in case the pose_np is not detected, the pose_np value would be too small, which would effect the
                # -- calculation of mean and std
                if abs(pose_np[i, 0, i_kpt] + save_pose_root[0]) < 5 and \
                        abs(pose_np[i, 1, i_kpt] + save_pose_root[1]) < 5:
                    continue
                weight = num[i, i_kpt] / (num[i, i_kpt] + 1)
                np_avg[i, :, i_kpt] = np_avg[i, :, i_kpt] * weight + (1 - weight) * (pose_np[i, :2, i_kpt])
                num[i, i_kpt] += 1
    return np_avg


def cal_std_parted(tuple_in):
    np_avg, df_pose, idx = tuple_in
    num = np.zeros((64, 137))
    np_var = np.zeros((64, 2, 137))
    for i_pose_fn in tqdm.tqdm(df_pose, desc="v # {}: ".format(idx), position=idx):
        pose_np = np.load(i_pose_fn)["pose"]
        for i in range(64):
            pose_np, save_pose_root = pose_np_deduct_root(pose_np, i)
            for i_kpt in range(137):

                # in case the pose_np is not detected, the pose_np value would be too small, which would effect the
                # -- calculation of mean and var
                if abs(pose_np[i, 0, i_kpt] + save_pose_root[0]) < 5 and \
                        abs(pose_np[i, 1, i_kpt] + save_pose_root[1]) < 5:
                    continue
                weight = num[i, i_kpt] / (num[i, i_kpt] + 1)
                np_var[i, :, i_kpt] = np_var[i, :, i_kpt] * weight + (1 - weight) * (
                        pose_np[i, :2, i_kpt] - np_avg[i, :2, i_kpt]) ** 2
                num[i, i_kpt] += 1
    return np.sqrt(np_var)
    


def global2local_parted(poses):
    """ Localize keypoints by separating the face and hands from the body.
    """
    # The following indices are define in pose-137.
    global_root = 1
    face_begin = 25
    handL_begin = 95
    handR_begin = 116

    face_root = face_begin + 30
    handL_root = 7
    handR_root = 4

    # make absolute coordinates relative to the root point
    poses[..., :2, :] = poses[..., :2, :] - poses[..., :2, global_root, None]

    # global to local
    # # face
    indices = list(range(face_begin, face_root)) + list(range(face_root + 1, face_begin + 70))
    poses[..., :2, indices] = poses[..., :2, indices] - poses[..., :2, face_root, None]

    # # hands
    poses[..., :2, handL_begin:handL_begin + 21] = poses[..., :2, handL_begin:handL_begin + 21] - \
                                                   poses[..., :2, handL_root, None]
    poses[..., :2, handR_begin:handR_begin + 21] = poses[..., :2, handR_begin:handR_begin + 21] - \
                                                   poses[..., :2, handR_root, None]

    return poses

def cal_mean_std():
    parser = argparse.ArgumentParser(description="argparser for cal_mean_std")
    # parser.add_argument("-f", type=str, help="file name to compute mean and std", default=None)
    parser.add_argument('-b', '--base_dataset_path', default=None, help="dataset root path", required=True)
    parser.add_argument('-s', '--speaker', default=None, help='Speaker Name', required=True)

    parser.add_argument("-np", "--num_processes", type=int, default=10, help="number of threads")

    parser.add_argument("-m", "--mode", default="parted", type=str,
                        help="switch which mode of skeleton architecture in use ('parted', 'global').")

    parser.add_argument("--mean", help="only calculate mean", action="store_true")
    parser.add_argument("--std", help="only calculate std", action="store_true")
    parser.add_argument("-d", "--debug", help="debug mode", action="store_true")
    args = parser.parse_args()

    # assert (args.f is not None) or (args.b is not None and args.b is not None)
    file_nm = os.path.join(args.base_dataset_path, args.speaker, "clips.csv")
    dataset_path = os.path.join(args.base_dataset_path, args.speaker)

    """ Switch which mode of skeleton architecture in use ('global', 'parted'). """
    assert args.mode == "global" or args.mode == "parted"
    if args.mode == "parted":
        # This is the default scenario, in which to use hierarchical pose.
        fn_cal_mean = cal_mean_parted
        fn_cal_std = cal_std_parted
        print("Using parted pose.")
    elif args.mode == "global":
        fn_cal_mean = cal_mean_global
        fn_cal_std = cal_std_global
        print("Using global pose.")

    file_nm_mean_std = f"mean_std-{args.mode}"

    fp_mean_npy = os.path.join(dataset_path, "tmp", f"mean_std-{args.mode}.npy")
    fp_mean_std_npz = os.path.join(dataset_path, f"{file_nm_mean_std}.npz")
    # fp_mean_std_json = os.path.join(dataset_path, f"{file_nm_mean_std}.json")

    print("\n=== Processsing {}".format(file_nm))

    df = pd.read_csv(os.path.join(file_nm))
    df_train = df[df["dataset"] == "train"]
    df_pose = df_train["pose_fn"]
    num_thread = args.num_processes
    stride = len(df_pose) // num_thread
    if args.debug:
        stride = 5
    lst_df_pose = [(df_pose[i * stride: (i + 1) * stride], i) for i in range(num_thread)]

    if not args.std:
        # calculate mean
        print("\n=== Calculating mean ...")
        p = Pool(num_thread, initializer=tqdm.tqdm.set_lock, initargs=(RLock(),))
        # ans = p.map(cal_mean_del_0, lst_df_pose)
        ans = p.map(fn_cal_mean, lst_df_pose)
        ans = np.array(ans)
        np_avg_save = np.expand_dims(np.average(np.average(ans, axis=0), axis=0), axis=0)
        if not args.debug:
            np.save(fp_mean_npy, np_avg_save)
    else:
        print("\n=== Using pre-calculated mean ...")
        np_avg_save = np.load(fp_mean_npy)

    if args.mean:
        exit()

    print("\n=== Calculating std ...")

    np_avg = np.array([np_avg_save.squeeze() for _ in range(64)])
    lst_df_pose = [(np_avg, df_pose[i * stride: (i + 1) * stride], i)
                   for i in range(num_thread)]
    freeze_support()
    p = Pool(num_thread, initializer=tqdm.tqdm.set_lock, initargs=(RLock(),))
    # ans = p.map(cal_std_del_0, lst_df_pose)
    ans = p.map(fn_cal_std, lst_df_pose)
    ans = np.array(ans)
    np_std_save = np.expand_dims(np.average(np.average(ans, axis=0), axis=0), axis=0)

    if not args.debug:
        np.savez(fp_mean_std_npz, mean=np_avg_save, std=np_std_save)

        # np_avg_save = np_avg_save.squeeze(0)
        # np_std_save = np_std_save.squeeze(0)
        # dct2save = {"mean": np_avg_save.tolist(), "std": np_std_save.tolist()}
        # with open(fp_mean_std_json, "w") as f:
        #     json.dump(dct2save, f)

        print(f"Deleting temporary files")
        # if os.path.exists(fp_mean_std_npz):
        #     os.remove(fp_mean_std_npz)
        if os.path.exists(fp_mean_npy):
            os.remove(fp_mean_npy)


if __name__ == "__main__":
    cal_mean_std()
