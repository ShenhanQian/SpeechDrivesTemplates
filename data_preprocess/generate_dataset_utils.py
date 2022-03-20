import numpy as np
import cv2
import os
import sys
import tqdm
# import ffmpeg
import pandas as pd
import pdb
from multiprocessing import Pool, RLock, freeze_support
import shutil


def dir_video2frames(video_dir, target_dir, fps=15):
    assert fps in [15, 25]
    print("====== FN dir_video2frames ======")
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    else:
        print(f"Warning: target directory {target_dir} exists!")
        # input("Please confirm by pressing any key, ABORT by pressing Ctrl+c")
    lst_videos = sorted(os.listdir(video_dir))
    for video_nm in tqdm.tqdm(lst_videos):
        print(f"Processing video: {os.path.join(video_dir, video_nm)}")
        video_nm_wo_ext = os.path.splitext(video_nm)[0]
        if not os.path.exists(os.path.join(target_dir, video_nm_wo_ext)):
            os.mkdir(os.path.join(target_dir, video_nm_wo_ext))
        command = f'ffmpeg -i {os.path.join(video_dir, video_nm)} -qscale 0 -r {fps} -y ' \
                  f'{os.path.join(target_dir, video_nm_wo_ext)}/{video_nm_wo_ext}_%6d.jpg'
        print(f'command: {command}')
        os.system(command)


def dir_change_fps(video_dir, target_dir):
    print("====== FN dir_change_fps ======")
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    else:
        print(f"Warning: target directory {target_dir} exists!")
        input("Please confirm by pressing any key, ABORT by pressing Ctrl+c")
    lst_videos = os.listdir(video_dir)
    for video_nm in tqdm.tqdm(lst_videos):
        print(f"change_fps video: {video_nm}")
        command = f"ffmpeg -i {os.path.join(video_dir, video_nm)} -qscale 0 -r 15 -y {os.path.join(target_dir, video_nm)}"
        print(f'command: {command}')
        os.system(command)


def dir_change_resolution(video_dir, target_dir):
    lst_videos = os.listdir(video_dir)
    for video_nm in tqdm.tqdm(lst_videos):
        print(f"change_fps video: {video_nm}")
        command = f"ffmpeg -i {os.path.join(video_dir, video_nm)} -qscale 0 -strict -2 -vf scale=-1:720 -y {os.path.join(target_dir, video_nm)}"
        print(f'command: {command}')
        os.system(command)


def show_file():
    # np_train = np.load("/group/speech2gesture/oliver/train_137/npz/215140-00:09:41.766667-00:09:45.966667.npz")
    # df = pd.read_csv("/group/projects/voice2pose/data/train_luoxiang_137_3.csv")
    # df = pd.read_csv("/group/speech2gesture/train_oliver_137_3.csv")
    # np_kpt = np.load("/group/projects/voice2pose/data/luoxiang/kp/BV1264y1c7e6/BV1264y1c7e6_03333.npy")
    luoxiang_npz = np.load("/group/projects/voice2pose/data/luoxiang/train_137_3/npz/BV1xT4y177Kq-996-1060.npz")
    # print(np_kpt.shape)
    # print(df.columns)

    # print(df.iloc[0])
    # print(df.iloc[1])
    pdb.set_trace()


def frame_idx_to_time(frame_idx, fps=15):
    all_seconds = frame_idx / float(fps)
    hour = int(all_seconds // 3600)
    minute = int((all_seconds % 3600) // 60)
    seconds = (all_seconds % 3600) % 60
    ans_time = f"{hour:02d}:{minute:02d}:{seconds:09.6f}"
    return ans_time


def check_file():
    dataset_root = "/group/projects/voice2pose/data/luoxiang/"
    vid_dir = os.path.join(dataset_root, "videos_15fps_720p")
    frame_dir = os.path.join(dataset_root, "frames_15fps_720p")
    kpt_dir = os.path.join(dataset_root, "kp")
    for vid_nm in sorted(os.listdir(frame_dir)):
        print(f'\n===== video: {vid_nm} =====')
        t_frame_dir = os.path.join(frame_dir, vid_nm)
        t_kpt_dir = os.path.join(kpt_dir, vid_nm)

        num_frame = len(os.listdir(t_frame_dir))
        num_kpt = len(os.listdir(t_kpt_dir))
        print(f'num_frame: {num_frame}')
        print(f'num_kpt: {num_kpt}')
        print(f'kpt lost: {num_frame - num_kpt}')

        for i in range(77, num_frame+1):
            frame_fn = f"{vid_nm}_{i:05d}.jpg"
            kpt_fn = f"{vid_nm}_{i:05d}.npy"
            kpt_path = os.path.join(t_kpt_dir, kpt_fn)
            frame_path = os.path.join(t_frame_dir, frame_fn)
            if not os.path.exists(kpt_path):
                print(i, frame_idx_to_time(i), end=",")
            if not os.path.exists(frame_path):
                print(f"frame: {i}")


def split_train_test():
    train_test_ratio = 0.9
    print(f"Fn ``split_train_test'', train_test_ratio={train_test_ratio}")
    idle_num = 13
    # Since when generating data samples, stride=5, then after at least 13 data samples,
    # the train and validation set would completely share no frames
    dr_all_csv = "/group/projects/voice2pose/data/intermediate_csv/train_v2"
    ls_all_csv = sorted(os.listdir(dr_all_csv))
    ls_all_csv = [os.path.join(dr_all_csv, i) for i in ls_all_csv]
    print(dr_all_csv)
    ls_train_df, ls_test_df, ls_idle_df = [], [], []
    for csv_path in ls_all_csv:
        df = pd.read_csv(csv_path)
        total_num = len(df)
        train_test_split = int(total_num*train_test_ratio)
        ls_train_df.append(df.iloc[:train_test_split])

        idle_df = df.iloc[train_test_split: train_test_split + idle_num]
        # change the ``dataset'' column in idle_df to be ``idle''
        idle_df.loc[:, "dataset"] = "idle"
        ls_idle_df.append(idle_df)

        test_df = df.iloc[train_test_split + idle_num:]
        # change the ``dataset'' column in test_df to be ``dev''
        test_df.loc[:, "dataset"] = "val"
        ls_test_df.append(test_df)

        print(f"file: {os.path.basename(csv_path)}, total_num: {total_num}, "
              f"train: {train_test_split}, test: {total_num - train_test_split}")
    ans_df = pd.concat([pd.concat(ls_train_df), pd.concat(ls_idle_df), pd.concat(ls_test_df)])
    ans_df.to_csv(os.path.join(dr_all_csv, f"train_luoxiang_137_3.csv"), index=False)


def cal_speaker_scalar(speaker=None, speaker_file_path=None):
    def _cal_shoulder_distance(np_pose):
        return np.sqrt(np.sum((np_pose[0, :, 2] - np_pose[0, :, 5]) ** 2))

    oliver_scalar = 0.9549234615419752
    oliver_shoulder_dist = 331.0850066245443

    data_root = "/group/speech2gesture/"
    luoxiang_data_root = "/group/projects/voice2pose/data"

    if speaker_file_path is not None:
        mean_std_file_path = speaker_file_path
    elif speaker is not None:
        mean_std_file_path = os.path.join(data_root, "mean_std", f"train_{speaker}_137_3_mean_std.npz")
        if not os.path.exists(mean_std_file_path):
            mean_std_file_path = os.path.join(data_root, "mean_std", f"train_{speaker}_137_mean_std.npz")
        if not os.path.exists(mean_std_file_path):
            mean_std_file_path = os.path.join(data_root, f"train_{speaker}_137_3_mean_std.npz")
        if not os.path.exists(mean_std_file_path):
            mean_std_file_path = os.path.join(data_root, f"train_{speaker}_137_mean_std.npz")
    else:
        print("ERROR! Please provde the speaker to calculate scalar.")
        exit(1)
    # print(f"{mean_std_file_path}")

    if mean_std_file_path.endswith(".npz"):
        np_data = np.load(mean_std_file_path)['mean']
    elif mean_std_file_path.endswith(".npy"):
        np_data = np.load(mean_std_file_path)
    else:
        print(f"ERROR file type: {mean_std_file_path}")
        exit(1)

    speaker_shoulder_distance = _cal_shoulder_distance(np_data)
    speaker_scalar = oliver_shoulder_dist * oliver_scalar / speaker_shoulder_distance
    print(f"speaker: {speaker if speaker is not None else os.path.split(speaker_file_path)[1]}, "
          f"scalar: {speaker_scalar}, speaker_shoulder_distance: {speaker_shoulder_distance}")


def distribute_for_multiprocess(lst_all, num_process):
    ans = []
    num_total = len(lst_all)
    num_iter = num_total // num_process
    if num_total % num_process > 0:
        num_iter += 1
    for i in range(num_process - 1):
        ans.append(lst_all[i * num_iter: (i+1) * num_iter])
    ans.append(lst_all[(num_process - 1) * num_iter:])
    return ans


def check_dataset_single(csv_path):
    speaker = os.path.split(csv_path)[1].split('_')[1]
    df = pd.read_csv(csv_path)
    df_train = df[df['dataset'] == 'train']
    df_dev = df[df['dataset'] == 'dev']
    df_idle = df[df['dataset'] == 'idle']
    # print(len(df[df['dataset'].apply(lambda x: x not in ['train', 'dev'])]))
    print(f'speaker: {speaker:>9s}, train: {len(df_train):>6d}, dev: {len(df_dev):>6d}, idle: {len(df_idle):>3d}, '
          f'total: {len(df):>6d}')


def dataset_statistics():
    def _get_csv_fn(speaker):
        dr_root = "/group/speech2gesture/"
        csv_fn = os.path.join(dr_root, f'train_{speaker}_137_3.csv')
        if not os.path.exists(csv_fn):
            csv_fn = os.path.join(dr_root, f'train_{speaker}_137.csv')
        return csv_fn

    lst_speaker = ['oliver', 'jon', 'luoxiang', 'lige', 'almaram',
                   'angelica', 'conan', 'ellen', 'chemistry', 'shelly', 'seth']
    lst_dataset = [_get_csv_fn(i) for i in lst_speaker]
    # lst_dataset = ["/group/speech2gesture/train_oliver_137_3.csv",
    #                "/group/speech2gesture/train_jon_137_3.csv",
    #                "/group/projects/voice2pose/data/train_luoxiang_v2_137_3.csv",
    #                "/group/projects/voice2pose/data/train_ligeV2_137_3.csv",
    #                ]
    for dataset in lst_dataset:
        try:
            check_dataset_single(dataset)
        except:
            print(f"ERROR when checking {dataset}")


class Speech2gestureDatasetGenerator:
    def __init__(self):
        pass

    @staticmethod
    def pose137_to_pose122(x):
        x = x.transpose(1, 0)
        return np.concatenate([x[0:8, :],  # upper_body
                               x[15:17, :],   # eyes
                               x[25:, :]], axis=0).transpose((1, 0))   # face, hand_l and hand_r

    def check_kp(self, lst_in):
        idx, base = lst_in
        outline_cnt = 0
        # print(base, "start")
        trash_dr = os.path.join(base, "../del_1-10")
        if not os.path.exists(trash_dr):
            os.mkdir(trash_dr)
        for kp_path in tqdm.tqdm(sorted(os.listdir(base)), desc=f"{idx}", position=idx):
            kp = np.load(os.path.join(base, kp_path))
            kp = self.pose137_to_pose122(kp)[:2, :]
            x_min = np.min(kp[0])
            x_max = np.max(kp[0])
            y_min = np.min(kp[1])
            y_max = np.max(kp[1])
            if x_min < 15 or x_max > 1280 - 15 or y_min < 5:
                # print(kp_path)
                outline_cnt += 1
                shutil.move(os.path.join(base, kp_path), trash_dr)
                # os.remove(os.path.join(base, kp_path))
                # print(x_min, max(1700,x_max))
                # canvas = render_luoxiang(kp)[:,380:380+1080,:]
                # canvas = cv2.resize(canvas,(512,512))
                # cv2.imshow("1",canvas)
                # cv2.waitKey(300)
        print(base, outline_cnt)

    def main_check_kp(self, is_multi_process=True):
        base = "/group/projects/voice2pose/data/ligeV2/kp_frames_15fps"

        lst_all = sorted(os.listdir(base))
        lst_all = [os.path.join(base, i) for i in lst_all]

        if is_multi_process:
            num_process = len(lst_all)
            input(f"Multi-Processing, num_process :{num_process}, press ENTER to confirm.")
            lst_distro = [[i, lst_all[i]] for i in range(num_process)]

            freeze_support()
            p = Pool(num_process, initializer=tqdm.tqdm.set_lock, initargs=(RLock(),))
            p.map(self.check_kp, lst_distro)
        else:
            for base in tqdm.tqdm(lst_all):
                self.check_kp([0, base])

    def check_kp_single(self, lst_in):
        idx, lst_kp, trash_dr = lst_in
        outline_cnt = 0
        for kp_path in tqdm.tqdm(lst_kp, desc=f"{idx}", position=idx):
            kp = np.load(kp_path)
            kp = self.pose137_to_pose122(kp)[:2, :]
            x_min = np.min(kp[0])
            x_max = np.max(kp[0])
            y_min = np.min(kp[1])
            y_max = np.max(kp[1])
            if x_min < 15 or x_max > 1280 - 15 or y_min < 5:
                outline_cnt += 1
                shutil.move(kp_path, trash_dr)

    def check_kp_video_mp(self, video_fn, num_process=64):
        """check_kp_video_multi-process"""
        dr_root = "/group/projects/voice2pose/data/ligeV2/kp_frames_15fps"
        trash_dr = os.path.join(dr_root, "del_1-10")

        video_dr = os.path.join(dr_root, video_fn)
        lst_all = sorted(os.listdir(video_dr))
        lst_all = [os.path.join(video_dr, i) for i in lst_all]

        input(f"Multi-Processing, num_process :{num_process}, press ENTER to confirm.")
        lst_all = distribute_for_multiprocess(lst_all, num_process)
        lst_distro = [[i, lst_all[i], trash_dr] for i in range(num_process)]

        freeze_support()
        p = Pool(num_process, initializer=tqdm.tqdm.set_lock, initargs=(RLock(),))
        p.map(self.check_kp_single, lst_distro)
        p.close()
        p.join()


class LigeDatasetGenerator:
    def __init__(self):
        self.data_root = "/group/projects/voice2pose/data/ligeV2"
        # self.left_scalar = 0.9204052623064795  # data without 1-10
        # self.right_scalar = 1.0569754082050022  # data without 1-10
        self.left_scalar = 0.9177568298452349
        self.right_scalar = 1.0680188893714462
        self.scalar_right_2_left = self.right_scalar / self.left_scalar

    @staticmethod
    def split_left_right_single(lst_in):
        p_idx, lst_fn = lst_in
        thd = 640

        ans = []
        for pose_fn in tqdm.tqdm(lst_fn, desc=f"Prs{p_idx}", position=p_idx):
            np_pose = np.load(pose_fn)
            pose_rt = np_pose[0, 1]
            camera_pos = "left" if pose_rt <= thd else "right"
            # print(f"pose_rt: {pose_rt}, camera_pos: {camera_pos}")

            ans.append({"pose_fn": pose_fn, "camera": camera_pos})

            # rename file
            f_dr, f_nm_raw = os.path.split(pose_fn)
            f_nm, f_nm_postfix = os.path.splitext(f_nm_raw)
            if f_nm.endswith("_l") or f_nm.endswith("_r"):
                continue
            f_nm = f_nm + "_l" if camera_pos == "left" else f_nm + "_r"
            f_path = os.path.join(f_dr, f_nm + f_nm_postfix)
            os.rename(pose_fn, f_path)
        df = pd.DataFrame(ans)
        return df

    def split_left_right(self, frame_dir="33", num_process=64, is_debug=False):
        print("This is FN ``split_left_right'' ")
        frame_dir_path = os.path.join(self.data_root, "kp_frames_15fps", frame_dir)
        lst_frame_all = sorted(os.listdir(frame_dir_path))
        lst_frame_all = [os.path.join(frame_dir_path, i) for i in lst_frame_all]

        if is_debug:
            self.split_left_right_single([0, lst_frame_all])
            exit()

        lst_distro_all = distribute_for_multiprocess(lst_frame_all, num_process)
        lst_distro = [[i, lst_distro_all[i]] for i in range(num_process)]

        # freeze_support()
        # p = Pool(num_process, initializer=tqdm.tqdm.set_lock, initargs=(RLock(),))
        p = Pool(num_process)
        lst_ans = p.map(self.split_left_right_single, lst_distro)
        ans = pd.concat(lst_ans)
        ans.to_csv(os.path.join(self.data_root, f"{frame_dir}.csv"), index=False)

        p.close()
        p.join()

    @classmethod
    def compare_shoulder(cls, csv_fn="/group/projects/voice2pose/data/ligeV2/36.csv"):
        print(f"This is FN ``LigeDatasetGenerator::compare_shoulder'', param: {csv_fn}")

        df = pd.read_csv(csv_fn)
        df_l = df[df["camera"] == "left"]
        df_r = df[df["camera"] == "right"]

        num = 0
        avg = 0
        for pose_fn in tqdm.tqdm(df_l["pose_fn"].to_list()):
            np_pose = np.load(pose_fn)
            weight = num / (num + 1)
            avg = avg * weight + (1 - weight) * np.sqrt(np.sum((np_pose[:, 2] - np_pose[:, 5]) ** 2))
            num += 1

        left_average = avg

        num = 0
        avg = 0
        for pose_fn in tqdm.tqdm(df_r["pose_fn"].to_list()):
            np_pose = np.load(pose_fn)
            weight = num / (num + 1)
            avg = avg * weight + (1 - weight) * np.sqrt(np.sum((np_pose[:, 2] - np_pose[:, 5]) ** 2))
            num += 1

        right_average = avg

        print(f"left: {left_average}, right: {right_average}")

    @classmethod
    def unify_left_right_direct(cls):
        pre_csv = "/group/projects/voice2pose/data/train_sep_ligeV2_137_3.csv"
        pre_path = "/group/projects/voice2pose/data/ligeV2/train/"
        uni_csv = "/group/projects/voice2pose/data/train_ligeV2_137_3.csv"
        uni_path = "/group/projects/voice2pose/data/ligeV2/train_uni/"
        left_scalar = 0.9204052623064795
        right_scalar = 1.0569754082050022

        scalar_right_2_left = right_scalar / left_scalar

        # # fix the path change
        if not os.path.exists(uni_csv):
            df = pd.read_csv(pre_csv)
            df["pose_fn"] = df["pose_fn"].apply(lambda x: x.replace(pre_path, uni_path))
            df.to_csv(uni_csv, index=False)

        df = pd.read_csv(uni_csv)

        dfp = df[df['camera'] == "right"]['pose_fn']
        for i in tqdm.tqdm(range(len(df))):
            pose_fn = dfp.iloc[i]
            pose_npz = np.load(pose_fn)

            pose_np = pose_npz['pose']
            img_fns = pose_npz['imgs']
            wav = pose_npz['audio']

            pose_np *= scalar_right_2_left
            np.savez(pose_fn, pose=pose_np, imgs=img_fns, audio=wav)

    def unify_left_right_single(self, lst_in):
        idx_process, df = lst_in
        idx_pose_fn = list(df.columns).index("pose_fn")
        idx_camera = list(df.columns).index("camera")
        for i in tqdm.tqdm(range(len(df)), desc=f"{idx_process}", position=idx_process):
            if df.iloc[i, idx_camera] != "right":
                continue
            pose_fn = df.iloc[i, idx_pose_fn]
            pose_npz = np.load(pose_fn)

            pose_np = pose_npz['pose']
            img_fns = pose_npz['imgs']
            wav = pose_npz['audio']

            pose_np *= self.scalar_right_2_left
            np.savez(pose_fn, pose=pose_np, imgs=img_fns, audio=wav)
            df.iloc[i, idx_camera] = "uni_right"
        return df

    def unify_left_right(self, num_process=64):
        pre_csv = "/group/projects/voice2pose/data/train_ligeV2_sep_137_3.csv"
        pre_path = "/group/projects/voice2pose/data/ligeV2/train/"
        uni_csv = "/group/projects/voice2pose/data/train_ligeV2_137_3.csv"
        uni_path = "/group/projects/voice2pose/data/ligeV2/train_uni/"

        # # fix the path change
        if not os.path.exists(uni_csv):
            df = pd.read_csv(pre_csv)
            df["pose_fn"] = df["pose_fn"].apply(lambda x: x.replace(pre_path, uni_path))
            df.to_csv(uni_csv, index=False)

        df = pd.read_csv(uni_csv)

        p = Pool(num_process)

        lst_distro = []
        num_total = len(df)
        num_iter = num_total // num_process
        if num_total % num_process > 0:
            num_iter += 1
        for i in range(num_process - 1):
            lst_distro.append([i, df.iloc[i * num_iter: (i + 1) * num_iter, :]])
        lst_distro.append([num_process - 1, df.iloc[(num_process - 1) * num_iter:, :]])

        ans = p.map(self.unify_left_right_single, lst_distro)
        p.close()
        p.join()

        print("All sub processes ends.")
        df = pd.concat(ans)
        df.to_csv(uni_csv, index=False)
        print("File saved")

    # @classmethod
    # def rename_file(cls, dr):
    #     lst_all = sorted(os.listdir(dr))
    #     lst_all = [os.path.join(dr, i) for i in lst_all]
    #     for pose_fn in lst_all:
    #         f_dr, f_nm_raw = os.path.split(pose_fn)
    #         f_nm, f_nm_postfix = os.path.splitext(f_nm_raw)
    #         f_nm = "_".join(f_nm.split('_')[:-1])
    #         print(f_nm)
    #         f_path = os.path.join(f_dr, f_nm + f_nm_postfix)
    #         os.rename(pose_fn, f_path)


if __name__ == "__main__":
    pass
    """LigeDataset"""
    # LigeDatasetGenerator().unify_left_right()
    # LigeDatasetGenerator.compare_shoulder("/group/projects/voice2pose/data/ligeV2/11.csv")
    # LigeDatasetGenerator.rename_file("/group/projects/voice2pose/data/ligeV2/kp_frames_15fps/15")
    # LigeDatasetGenerator().split_left_right(frame_dir="1-10")

    """Speech2gestureDataset"""
    # Speech2gestureDatasetGenerator().check_kp([0, "/group/projects/voice2pose/data/ligeV2/kp_frames_15fps/1-10"])
    # Speech2gestureDatasetGenerator().check_kp_video_mp("1-10")
    # Speech2gestureDatasetGenerator().main_check_kp()

    """Perform on All Dataset"""
    # dir_change_resolution('/group/projects/voice2pose/data/luoxiang/videos_15fps',
    #                       '/group/projects/voice2pose/data/luoxiang/videos_15fps_720p')

    # dir_change_fps('/group/projects/voice2pose/data/006/006_videos',
    #                '/group/projects/voice2pose/data/006/006_videos_15fps')

    dir_video2frames(video_dir='/group/projects/voice2pose/data/006/006_videos_15fps',
                     target_dir='/group/projects/voice2pose/data/006/frames_15fps',)
    #
    # rm_files("/group/projects/voice2pose/data/luoxiang/frames")
    # show_file()
    # check_file()
    # split_train_test()
    # dataset_statistics()
    # cal_speaker_scalar(speaker="oliver")
    # cal_speaker_scalar(speaker_file_path="/group/projects/voice2pose/data/train_ligeV2_sep_137_3_left_mean.npy")
    # cal_speaker_scalar(speaker_file_path="/group/projects/voice2pose/data/train_ligeV2_sep_137_3_right_mean.npy")
