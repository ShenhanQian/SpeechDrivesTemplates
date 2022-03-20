from generate_dataset_utils import dir_video2frames
import argparse
import os


parser = argparse.ArgumentParser(description='video to frames')
parser.add_argument('-b', '--base_dataset_path', type=str, default=None, help="dataset root path", required=True)
parser.add_argument('-s', '--speaker', type=str, default='Default Speaker Name', required=True)

parser.add_argument('-fps', type=int, default=15, help="Frame rate to extract frames from videos.")

args = parser.parse_args()

DATASET_PATH = os.path.join(args.base_dataset_path, args.speaker)
DIR_VIDEO_PATH = os.path.join(DATASET_PATH, "videos")
DIR_FRAME_PATH = os.path.join(DATASET_PATH, "frames")


if __name__ == "__main__":
    dir_video2frames(video_dir=DIR_VIDEO_PATH, target_dir=DIR_FRAME_PATH, fps=args.fps)
