# run openpose to extract 2D keypoints

import sys
import cv2
import os
from sys import platform
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["face"] = True
params["hand"] = True

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Image
def get_keypoints(frame_dir,folder,img_path,pose_dir):
    filename, file_extension = os.path.splitext(img_path)
    imageToProcess = os.path.join(frame_dir,folder,img_path)
    npy_store_path = os.path.join(pose_dir,folder,filename)

    datum = op.Datum()
    imageToProcess = cv2.imread(imageToProcess)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    if datum.poseKeypoints.shape == (1, 25, 3) and datum.faceKeypoints.shape == (1, 70, 3) and datum.handKeypoints[0].shape == (1, 21, 3) and datum.handKeypoints[1].shape == (1, 21, 3):
        # only collect frames with complete pose predictions
        
        npy = np.concatenate([datum.poseKeypoints,datum.faceKeypoints,datum.handKeypoints[0],datum.handKeypoints[1]],axis=1).squeeze()
        npy = npy.transpose(1,0)
        np.save(npy_store_path,npy)


if __name__=="__main__":
    """
    The folders are organized as follows
    {base}
    └── frames
    |   ├── vid_1
    |   │   ├── 000001.jpg
    |   │   ├── 000002.jpg
    |   |   └── xxxxxx.jpg
    |   └── vid_N
    |       ├── 000001.jpg
    |       ├── 000002.jpg
    |       └── xxxxxx.jpg
    └── tmp
        └── raw_pose_2d
            ├── vid_1
            │   ├── 000001.npy
            │   ├── 000002.npy
            |   └── xxxxxx.npy
            └── vid_N
                ├── 000001.npy
                ├── 000002.npy
                └── xxxxxx.npy
    """
    
    assert len(sys.argv) == 2
    base_dir = sys.argv[1]

    frame_dir = f"{base_dir}/frames/"
    pose_dir = f"{base_dir}/tmp/raw_pose_2d/"
    cnt = 0
    for i in os.listdir(frame_dir):
        if not os.path.exists(os.path.join(pose_dir,i)):
            os.makedirs(os.path.join(pose_dir,i))
        for j in os.listdir(os.path.join(frame_dir, i)):
            if j.endswith(".jpg"):
                if not os.path.exists(os.path.join(pose_dir,i,j.split(".")[0]+'.npy')):
                    cnt+=1
                    get_keypoints(frame_dir,i,j,pose_dir)
    print(cnt)           
    print("done!")
