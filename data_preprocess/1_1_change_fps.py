"""
python 1_1_change_fps.py <Directory_containing_original_videos> <Directory_of_output_videos>
"""
from generate_dataset_utils import dir_change_fps
import sys


if __name__ == "__main__":
    dir_change_fps(video_dir=sys.argv[1], target_dir=sys.argv[2])
