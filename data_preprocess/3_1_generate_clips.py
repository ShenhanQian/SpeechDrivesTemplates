import argparse
from multiprocessing import Pool, RLock, freeze_support
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import subprocess


# ## Constants
FRAMES_PER_SAMPLE = 64
FPS = 15
SR = 16000
LST_VIDEO_SUFFIX = [".mp4", ".MP4"]


parser = argparse.ArgumentParser(description='Extract data for the specified speaker')
parser.add_argument('-b', '--base_dataset_path', default=None, help="dataset root path", required=True)
parser.add_argument('-s', '--speaker', default='Default Speaker Name', required=True)

parser.add_argument('-np', '--num_processes', type=int, default=1)
parser.add_argument('--restart', action="store_true", help="By default this code would resume from last execution,"
                                                           "using existing csv files. Use argument --restart if want"
                                                           "to start the generate_clips process from the beginning.")

# argument to be set by user
parser.add_argument('-fi', '--start_frame_idx', type=int, default=80, help="The script will desert all the frames "
                                                                           "before start_frame_idx, because some video"
                                                                           "may have an introduction part that is not"
                                                                           "relevant to our task.")
# arguments do not need to change
parser.add_argument('-nf', '--num_frames', type=int, default=FRAMES_PER_SAMPLE)
parser.add_argument("-d", "--debug", help="debug mode", action="store_true")
args = parser.parse_args()

NUM_FRAMES = args.num_frames  # num frames per npz ( per sample)

# Parameters to be set here
DATA_DIR_NAME = "clips"
FRAME_DIR_NAME = "frames"
POSE_DIR_NAME = "rescaled_pose_2d"
VIDEO_DIR_NAME = "videos"

DATASET_PATH = os.path.join(args.base_dataset_path, args.speaker)  # dir containing videos, frames, key-points
FRAME_DIR_PATH = os.path.join(DATASET_PATH, FRAME_DIR_NAME)
VIDEO_DIR_PATH = os.path.join(DATASET_PATH, VIDEO_DIR_NAME)
POSE_DIR_PATH = os.path.join(DATASET_PATH, "tmp", POSE_DIR_NAME)
TMPCSV_DIR_PATH = os.path.join(DATASET_PATH, "tmp", "intermediate_csv")
if args.restart and os.path.exists(TMPCSV_DIR_PATH):
    print(f"Deleting previous temporary csv files.")
    ls_previous_tmpcsv = os.listdir(TMPCSV_DIR_PATH)
    for previous_tmpcsv in ls_previous_tmpcsv:
        os.remove(os.path.join(TMPCSV_DIR_PATH, previous_tmpcsv))
if not os.path.exists(TMPCSV_DIR_PATH):
    os.mkdir(TMPCSV_DIR_PATH)

assert os.path.exists(FRAME_DIR_PATH)
assert os.path.exists(VIDEO_DIR_PATH)
assert os.path.exists(POSE_DIR_PATH)

START_FRAME_IDX = args.start_frame_idx

print(f"Please confirm the paths: \nDATA_DIR_NAME: {DATA_DIR_NAME} \nFRAME_DIR_NAME: {FRAME_DIR_NAME} "
      f"\nPOSE_DIR_NAME: {POSE_DIR_NAME} \nVIDEO_DIR_NAME: {VIDEO_DIR_NAME} \nSTART_FRAME_IDX: {START_FRAME_IDX}")
# input("Press any key to confirm, else press Ctrl+c to abort.")

AUDIO_FN_TEMPLATE = os.path.join(args.base_dataset_path, f'%s/{DATA_DIR_NAME}/audio/%s-%s-%s.wav')
TRAINING_SAMPLE_FN_TEMPLATE = os.path.join(args.base_dataset_path, f'%s/{DATA_DIR_NAME}/npz/%s-%s-%s.npz')
if not os.path.exists(os.path.join(args.base_dataset_path, args.speaker, f"{DATA_DIR_NAME}/audio")):
    os.makedirs(os.path.join(args.base_dataset_path, args.speaker, f"{DATA_DIR_NAME}/audio"))
if not os.path.exists(os.path.join(args.base_dataset_path, args.speaker, f"{DATA_DIR_NAME}/npz")):
    os.makedirs(os.path.join(args.base_dataset_path, args.speaker, f"{DATA_DIR_NAME}/npz"))


def save_audio_sample_from_video(vid_path, audio_out_path, audio_start, audio_end, sr=48000):
    # in speech2gesture sr=44100
    if not (os.path.exists(os.path.dirname(audio_out_path))):
        try:
            os.makedirs(os.path.dirname(audio_out_path))
        except FileExistsError:
            print("dir has been created in another process")
    # cmd = 'ffmpeg -i "%s" -ss %s -to %s -ab 160k -ac 2 -ar %s -vn "%s" -y -loglevel warning' % (
    # vid_path, audio_start, audio_end, sr, audio_out_path)
    cmd = f'ffmpeg -i "{vid_path}" -ss {audio_start} -to {audio_end} ' \
          f'-ab 160k -ac 2 -ar {sr} -vn "{audio_out_path}" -y -loglevel warning'
    subprocess.call(cmd, shell=True)


def frame_idx_to_time(frame_idx):
    fps = FPS
    all_seconds = frame_idx / float(fps)
    hour = int(all_seconds // 3600)
    minute = int((all_seconds % 3600) // 60)
    seconds = (all_seconds % 3600) % 60
    ans_time = f"{hour:02d}:{minute:02d}:{seconds:09.6f}"
    return ans_time


def get_pose_path(frame_idx, video_nm):
    pose_dr = os.path.join(POSE_DIR_PATH, video_nm)
    pose_fn = video_nm + f"_{str(frame_idx).zfill(6)}.npy"
    pose_path = os.path.join(pose_dr, pose_fn)
    return pose_path


def get_pose_np(pose_path):
    """pose_np.shape = [3, 137]"""
    pose_np = np.load(pose_path)
    return pose_np


def get_frame_path(frame_idx, video_nm):
    frame_dr = os.path.join(FRAME_DIR_PATH, video_nm)
    frame_fn = video_nm + f"_{str(frame_idx).zfill(6)}.jpg"
    frame_path = os.path.join(frame_dr, frame_fn)
    frame_path.encode(encoding='utf-8')
    return frame_path


def get_video_path(video_nm):
    for suffix in LST_VIDEO_SUFFIX:
        video_path = os.path.join(VIDEO_DIR_PATH, video_nm + suffix)
        if os.path.exists(video_path):
            return video_path
    raise Exception


def gen_data_samples(dict_args):
    start_frame_idx = dict_args["start_frame_idx"]
    total_length = dict_args["total_length"]
    video_nm = dict_args["video_nm"]
    process_idx = dict_args["process_idx"]
    print(f"This is ``gen_data_samples'', "
          f"args are: start_frame_idx:{start_frame_idx}, "
          f"total_length:{total_length}, "
          f"video_nm:{video_nm}, ")
    if os.path.exists(os.path.join(TMPCSV_DIR_PATH, f"tmp_{video_nm}.csv")):
        return pd.read_csv(os.path.join(TMPCSV_DIR_PATH, f"tmp_{video_nm}.csv"))
    data_dict = {'dataset': [], 'start': [], 'end': [], 'interval_id': [],
                 'pose_fn': [], 'audio_fn': [], 'video_fn': [], 'speaker': []}

    video_fn = video_nm

    speaker_name = args.speaker

    # # poses.shape = [interval_length, 3, 137]
    # poses_interval = np.array([
    #     get_pose_np(get_pose_path(i, video_nm)) for i in range(start_frame_idx, total_length)
    # ])

    # img_fns_interval = np.array([
    #     get_frame_path(i, video_nm) for i in range(start_frame_idx, total_length)
    # ])

    end_frame_idx = total_length

    interval_start = frame_idx_to_time(start_frame_idx)
    interval_end = frame_idx_to_time(end_frame_idx)

    audio_out_path = AUDIO_FN_TEMPLATE % (speaker_name, video_nm, interval_start, interval_end)
    save_audio_sample_from_video(get_video_path(video_nm), audio_out_path, interval_start, interval_end)

    interval_start = pd.to_timedelta(frame_idx_to_time(start_frame_idx))

    interval_audio_wav, sr = librosa.load(audio_out_path, sr=SR, mono=True)
    sample_time_step = FPS // 3
    for frame_idx in tqdm(range(start_frame_idx, total_length - NUM_FRAMES, sample_time_step),
                          desc=f"video: {video_nm}", position=process_idx):
        try:
            # sample = df_interval[idx:idx + NUM_FRAMES]
            audio_start = (pd.to_timedelta(frame_idx_to_time(frame_idx)) - interval_start).total_seconds() * SR
            audio_end = (pd.to_timedelta(
                frame_idx_to_time(frame_idx + NUM_FRAMES)) - interval_start).total_seconds() * SR
            wav = interval_audio_wav[int(audio_start): int(audio_end)]

            # frames_out_path is the path to the npz file containing one single sample
            frames_out_path = TRAINING_SAMPLE_FN_TEMPLATE % (speaker_name, video_nm, frame_idx, frame_idx + NUM_FRAMES)
            if not (os.path.exists(os.path.dirname(frames_out_path))):
                os.makedirs(os.path.dirname(frames_out_path))

            """ Use online poses_np and img_fns"""
            poses_np = np.array([
                get_pose_np(get_pose_path(frame_idx + i, video_nm)) for i in range(NUM_FRAMES)
            ])

            img_fns = np.array([
                get_frame_path(frame_idx + i, video_nm) for i in range(NUM_FRAMES)
            ])
            np.savez(frames_out_path, pose=poses_np, imgs=img_fns, audio=wav)

            """ Use offline pose and imgs"""
            # ## pose: np.array of shape [64, 3, 137]
            # ## imgs: b str, path to frames
            # ## audio: np.array of shape (67200, )
            # np.savez(frames_out_path,
            #          pose=poses_interval[frame_idx-start_frame_idx: frame_idx-start_frame_idx+NUM_FRAMES],
            #          imgs=img_fns_interval[frame_idx-start_frame_idx: frame_idx-start_frame_idx+NUM_FRAMES],
            #          audio=wav)

            data_dict["dataset"].append("train")
            data_dict["start"].append(frame_idx)
            # data_dict["start"].append(sample.iloc[0]['pose_dt'])
            data_dict["end"].append(frame_idx + NUM_FRAMES)
            # data_dict["end"].append(sample.iloc[-1]['pose_dt'])
            data_dict["interval_id"].append(video_nm)
            # data_dict["interval_id"].append(interval)
            data_dict["pose_fn"].append(frames_out_path)
            data_dict["audio_fn"].append(audio_out_path)
            data_dict["video_fn"].append(video_fn)
            data_dict["speaker"].append(speaker_name)
        except:
            if args.debug:
                print(f'ERROR! video: {video_nm}\n')
            continue

        if args.debug:
            break

    ans_pd = pd.DataFrame.from_dict(data_dict)
    ans_pd.to_csv(os.path.join(TMPCSV_DIR_PATH, f"tmp_{video_nm}.csv"), index=False)
    # return ans_pd


if __name__ == "__main__":
    # vid_frame_dir = "/group/projects/voice2pose/data/luoxiang/frames_15fps_720p/BV1264y1c7e6"
    # total_length = len(os.listdir(vid_frame_dir))
    # gen_data_samples(start_frame_idx=76, total_length=total_length, video_nm="BV1264y1c7e6")

    ls_vid = os.listdir(FRAME_DIR_PATH)
    # start_frame_idx, total_length, video_nm
    ls_args = [{"video_nm": i,
                "total_length": len(os.listdir(os.path.join(FRAME_DIR_PATH, i))),
                "start_frame_idx": START_FRAME_IDX,
                "process_idx": idx
                } for idx, i in enumerate(ls_vid)]

    ls_args = np.array(ls_args)
    if args.num_processes > 1:
        freeze_support()
        p = Pool(min(args.num_processes, len(ls_args)), initializer=tqdm.set_lock, initargs=(RLock(),))
        # p = Pool(args.num_processes)
        dfs = p.map(gen_data_samples, ls_args)
    else:
        dfs = map(gen_data_samples, ls_args)
    # pd.concat(dfs).to_csv(os.path.join(DATASET_PATH, "clips.csv".format(args.speaker)), index=False)
    print(f"Clips for each video generated. To split train and validation set")
