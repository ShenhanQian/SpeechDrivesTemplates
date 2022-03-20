import os
import sys
import numpy as np
import cv2

from core.utils.keypoint_visualization import draw_body_parts


def formatted_print(digits):
    for i, val in enumerate(digits):
            print(val, end=', ')
            if i % 10 == 9:
                print()

def parsing_npz_137_mean_std(npz_path):
    delete_idx = [1] + list(range(8,15)) + list(range(17, 25))  # from 137 to 121

    items = np.load(npz_path, allow_pickle=True)
    mean = items['mean']
    std = items['std']

    mean = np.delete(mean, delete_idx, axis=2)
    std = np.delete(std, delete_idx, axis=2)

    print('\nmean:', mean.shape)
    formatted_print(list(mean.reshape(-1)))
    print('\nstd:', std.shape)
    formatted_print(list(std.reshape(-1)))
    
    print('\n')
    return mean, std

def vis_mean_pose(mean):
    W = 1280
    H = 1280//16*9

    img = np.zeros([H, W, 3], dtype=np.uint8) + 240
    mean = mean[0].transpose(1, 0)
    
    # head
    mean[9:39, :] += mean[39:40, :]
    mean[40:79, :] += mean[39:40, :]

    # hand
    mean[79:100, :] += mean[6:7, :]
    mean[100:121, :] += mean[3:4, :]

    mean[:, 0] += W // 2
    mean[:, 1] += H // 2

    draw_body_parts(img, mean)
    cv2.imshow('0', img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    assert len(sys.argv) == 2
    npz_path = sys.argv[1]
    assert os.path.exists(npz_path)
    
    mean, std = parsing_npz_137_mean_std(npz_path)
    # vis_mean_pose(mean)


