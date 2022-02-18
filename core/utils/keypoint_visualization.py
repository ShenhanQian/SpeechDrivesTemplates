import numpy as np
import cv2
import os
from tqdm import tqdm
import csv


def draw_landmarks(img, kps, color=(0,255,0), size=2, show_idx=False, edge_list=None, with_confidence_score=False):
    if edge_list is not None:
        for e in edge_list:
            cv2.line(img, (int(kps[e][0][0]), int(kps[e][0][1])), (int(kps[e][1][0]), int(kps[e][1][1])), color, size, cv2.LINE_AA)
    # for idx in range(kps.shape[0]):
    #     x = int(kps[idx][0])
    #     y = int(kps[idx][1])
    #     if with_confidence_score:
    #         if kps[idx].shape[0] != 3:
    #             print('cannot find keypoint confidence!')
    #             import ipdb; ipdb.set_trace()
    #         pt_color = int(kps[idx][2] * 255)
    #         if pt_color < 125:
    #             continue
    #         pt_color = (256-pt_color, pt_color, pt_color)
    #     else:
    #         pt_color = color
    #     if not show_idx:
    #         cv2.circle(img, (x, y), size-1, pt_color, -1)
    #     else:
    #         cv2.putText(img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pt_color)
    return img

def draw_hand_landmarks(img, kps, color=(255,0,0), size=2, show_idx=False, edges_list=None):
    if edges_list is not None:
        for idx, edges in enumerate(edges_list):
            for e in edges:
                color_lvl = 255 / 8 * (idx+3)
                pt_color =  (255, color_lvl, 1-color_lvl)
                cv2.line(img, (int(kps[e][0][0]), int(kps[e][0][1])), (int(kps[e][1][0]), int(kps[e][1][1])), pt_color, size, cv2.LINE_AA)
    # for idx in range(kps.shape[0]):
    #     x = int(kps[idx][0])
    #     y = int(kps[idx][1])
    #     if not show_idx:
    #         cv2.circle(img, (x, y), size, color, -1)
    #     else:
    #         cv2.putText(img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return img

def draw_body_parts(img, landmarks, size=2, with_confidence_score=False):
    num_keypoints = landmarks.shape[0]
    if num_keypoints == 135:
        num_pose=23
        num_hand=21
        num_face=70
        pose_edges = [[0,1], [0,4], [1,2], [4,5], [2,3], [5,6]]
    elif num_keypoints == 137:
        num_pose=25
        num_hand=21
        num_face=70
        pose_edges = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7]]
    elif num_keypoints == 121:
        num_pose=9
        num_hand=21
        num_face=70
        pose_edges = [[1,4], [1,2], [2,3], [4,5], [5,6]]
    else:
        raise NotImplementedError('Unsupported number of keypoints: %d' % num_keypoints)
    pose = landmarks[:num_pose]
    face = landmarks[num_pose:num_pose+num_face]
    hand_left = landmarks[num_pose+num_face:num_pose+num_face+num_hand]
    hand_right = landmarks[num_pose+num_face+num_hand:num_pose+num_face+num_hand*2]
        
    hand_edges = [
        [[0,1], [1,2], [2,3], [3,4]],
        [[0,5], [5,6], [6,7], [7,8]], 
        [[0,9], [9,10], [10,11], [11,12]], 
        [[0,13], [13,14], [14,15], [15,16]],
        [[0,17], [17,18], [18,19], [19,20]]]

    face_edges = [
        [0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9], [9,10], [10,11], [11,12], [12,13], [13,14], [14,15], [15,16],
        [17,18], [18,19], [19,20], [20,21],
        [22,23], [23,24], [24,25], [25,26], 
        [27,28], [28,29], [29,30],
        [31,32], [32,33], [33,34], [34,35], 
        [36,37], [37,38], [38,39], [39,40], [40,41], [41,36],
        [42,43], [43,44], [44,45], [45,46], [46,47], [47,42],
        [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], [55,56], [56,57], [57,58], [58,59], [59,48],
        [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,60],]

    draw_landmarks(img, pose, color=(25,175,25), size=size+2, show_idx=False, edge_list=pose_edges)
    # draw_landmarks(img, pose, color=(0,0,0), size=size, show_idx=False, edge_list=pose_edges)
    draw_landmarks(img, face, color=(100,100,100), size=size, show_idx=False, edge_list=face_edges)
    draw_hand_landmarks(img, hand_left, color=(255,0,0), size=size+1, show_idx=False, edges_list=hand_edges)
    draw_hand_landmarks(img, hand_right, color=(255,0,0), size=size+1, show_idx=False, edges_list=hand_edges)
    return img

def draw_pose_frames_in_long_img(poses):
    N = poses.shape[0]
    interval = 8
    poses = poses[:N-N%interval+1, :]
    N = poses.shape[0] // interval + 1
    H = 720
    w = H//3*4
    pose_step = H*0.7
    W = w + int((N-1) * pose_step)
    canvas = np.zeros([H, W, 3], dtype=np.uint8) + 255
    for i in range(poses.shape[0]):
        if i%interval == 0:
            window = canvas[:, int(i//interval*pose_step):int(i//interval*pose_step)+w, :]
            draw_pose(window, poses[i], np.array([[w//2, H//2]]))
    return canvas

def draw_pose(canvas, pose, center):
    pose = pose + center
    draw_body_parts(canvas, pose)

def vis_train_npz(npz_path, align_image=True, with_confidence_score=False):
    try:
        items = np.load(npz_path, allow_pickle=True)
    except:
        import ipdb; ipdb.set_trace()
        raise FileNotFoundError('Fail to load: %s' % npz_path)
    imgs = items['imgs']
    poses = items['pose']
    audio = items['audio']

    num_frames = len(imgs)
    if np.ndim(poses) not in [1, 3]:
        import ipdb; ipdb.set_trace()
    key = None
    for j, (img, pose) in tqdm(enumerate(zip(imgs, poses)), total=num_frames, desc='frames'):
        if align_image:
            img = str(img, 'utf-8')
            img = cv2.imread(img)
        else:
            img = np.zeros((720, 1280, 3)).astype(np.uint8) + 240
        try:
            img = draw_body_parts(img, pose.transpose(1,0), size=2, with_confidence_score=with_confidence_score)
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
            raise RuntimeError('Error in landmark ploting.')

        cv2.imshow('0', img)
        key = cv2.waitKey(60)
        if key == ord('q'):
            break
    return key

def vis_train_csv_dir(data_root, speaker, num_kp, align_image=True, with_confidence_score=False):
    csv_path = os.path.join(data_root, 'train_%s_%s_3.csv' %(speaker, str(num_kp)))
    with open(csv_path, newline='') as csvfile:
        rows = list(csv.reader(csvfile, delimiter=' ', quotechar='|'))
        column_titles = rows[0][0].split(',')

        for row in tqdm(rows[1:], desc='clips'):
            csv_items = {column_titles[i]: row[0].split(',')[i] for i in range(len(column_titles))}
            npz_path = csv_items['pose_fn']
            key = vis_train_npz(npz_path, align_image, with_confidence_score=with_confidence_score)
            if key == ord('q'):
                break
            
def vis_pose_npy(npy_path):
    pose = np.load(npy_path)
    img = np.zeros((720, 1280, 3)).astype(np.uint8)
    try:
        img = draw_landmarks(img, pose, (0, 255, 0), 4, True)
    except:
        import ipdb; ipdb.set_trace()
    cv2.imshow('0', img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

def translate_landmarks(kp, x, y):
    kp = kp + np.array([[x, y]])
    return kp

def vis_relative_pose(relative_pose, canvas_size, ):
    height, width = canvas_size

    img = np.zeros([height, width, 3]).astype(np.uint8) + 255
    translated_pose = translate_landmarks(relative_pose.transpose(1, 0), width//2, height//2)
    img = draw_body_parts(img, translated_pose)
    return img

def vis_relative_pose_clip(relative_poses, canvas_size):
    img_list = []
    for relative_pose in relative_poses:
        img = vis_relative_pose(relative_pose, canvas_size)
        img_list.append(img)
    return np.array(img_list)

def vis_relative_pose_pair(relative_pose_pred, relative_pose_gt, canvas_size):
    height, width = canvas_size

    img = np.zeros([height, width, 3]).astype(np.uint8) + 255
    translated_pose_pred = translate_landmarks(relative_pose_pred.transpose(1, 0), int(width*0.33), height//2)
    translated_pose_gt = translate_landmarks(relative_pose_gt.transpose(1, 0), int(width*0.67), height//2)
    img = draw_body_parts(img, translated_pose_pred)
    img = draw_body_parts(img, translated_pose_gt)
    return img

def vis_relative_pose_pair_clip(relative_poses_pred, relative_poses_gt, canvas_size):
    img_list = []
    for relative_pose_pred, relative_pose_gt in zip(relative_poses_pred, relative_poses_gt):
        img = vis_relative_pose_pair(relative_pose_pred, relative_pose_gt, canvas_size)
        img_list.append(img)
    return np.array(img_list)


if __name__ == "__main__":
    dataset_root = 'datasets/speakers/'
    speaker = 'oliver'
    num_kp = 137

    vis_train_csv_dir(dataset_root, speaker, num_kp, align_image=False, with_confidence_score=True)
