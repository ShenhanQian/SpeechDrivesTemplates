# Speech Drives Templates
The official repo for the ICCV-2021 paper "Speech Drives Templates: Co-Speech Gesture Synthesis with Learned Templates". <br>
[paper](https://arxiv.org/abs/2108.08020) / [video](https://youtu.be/yu-5gUHn6h8)

<p align="center">
  <img src="./iccv2021_sdt.jpg" width=500px/>
</p>

Our paper and this repo focus on upper-body pose generation from audio. To synthesize images from poses, please refer to this [Pose2Img](https://github.com/zyhbili/Pose2Img) repo.

**🔔 Update:**

- 2022-04-29: Upload checkpoints for all subjects.
- 2022-04-26: Change `POSE2POSE.LAMBDA_KL` in `config/default.py` from 1.0 to 0.1.

## Directory hierarchy

```
|-- config
|     |-- default.py
|     |-- voice2pose_s2g.yaml        # baseline: speech2gesture
|     |-- voice2pose_sdt_bp.yaml     # ours (Backprop)
|     |-- voice2pose_sdt_vae.yaml    # ours (VAE)
|     \-- pose2pose.yaml             # gesture reconstruction  
|
|-- core
|     |-- datasets
|     |-- netowrks
|     |-- pipelines
|     \-- utils
|
|-- datasets
|     \-- speakers
|           |-- oliver
|           |-- kubinec
|           \-- ...
|
|-- output
|     \-- <date-config-tag>  # A directory for each experiment
|
`-- main.py

```

## Installation

To generate videos, you need `ffmpeg` in your system.

```shell
sudo apt install ffmpeg
```

Install Python packages

```shell
pip install -r requirements.txt
```

## Dataset

We use a subset (Oliver and Kubinec) of the [Speech2Gesture](https://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/index.html) dataset and remove frames with bad human poses. We also collect data of two mandarine speakers (Luo and Xing).

To ease later research, we pack our processed data including **2d human pose sequences** and corresponding **audio clips**.
Please download from this [link](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/qianshh_shanghaitech_edu_cn/EhOVnrnCYS5KqDIkamXBJbgBLOzu8vEFGwy88jSRSNATFA?e=Hc0cOO) and organize the data under `datasets/speakers` as the above dirctory hierarchy.

Note that you do NOT need the **source video** frames to run this repo. In case you still want them for your own usage:
- For Luo and Xing, we provide the links of source videos as text files along side the above data packs.
- For Oliver and Kubinec, please refer to the [Speech2Gesture](https://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/index.html) dataset. 

Since our method address the entire upper body including the face and hands, the number of keypoints in our data is 137. For more details, please refer to [this](./pose_definition.md) document.

### Custom dataset
To build a dataset from custom videos, we provide reference scripts in `data_preprocess/`:
```
# ==== video processing ====
1_1_change_fps.py           # we use fps=15 by default
1_2_video2frames.py         # save each video as images

# ==== keypoint processing ====
2_1_gen_kpts.py             # use openpose to obtain keypoints
2_2_remove_outlier.py       # remove a frame with bad predicted keypoints
(2_3_rescale_shoulder_width.py  # rescale the keypoints)

# ==== npz processing ====
3_1_generate_clips.py       # generate a csv files as an index and npz files for clips
3_2_split_train_val_test.py # edit the csv file for dataset division

# ==== speakers_stat processing ====
4_1_calculate_mean_std.py   # save the mean and std of each keypoint (137 points) into a npy file
4_2_parse_mean_std_npz.py   # parse the above npy and print out for `speakers_stat.py`
```

> The step 2_3 is optional. It rescales the keypoints so that a new speaker has the same shoulder width as Oliver, and then you can simply copy the `scale_factor`  of Oliver for the new speaker in `speakers_stat.py`.

## Training SDT-BP

**Training** from scratch

``` bash
python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --tag oliver \
    DATASET.SPEAKER oliver \
    SYS.NUM_WORKERS 32
```

- `--tag` set the name of the experiment which wil be displayed in the outputfile.
- You can overwrite any parameter defined in `configs/default.py` by simply
adding it at the end of the command. The example above set `SYS.NUM_WORKERS` to 32 temporarily.

Resume **training** from an interrupted experiment

``` bash
python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --resume_from <checkpoint-to-continue-from> \
    DATASET.SPEAKER oliver
```

- With `--resume_from`, the program will load the `state_dict` from the checkpoint for both the model and the optimizer, and write results to the original directory that the checkpoint lies in.

**Training** from a pretrained model

``` bash
python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --pretrain_from <checkpoint-to-pretrain-from> \
    --tag oliver \
    DATASET.SPEAKER oliver
```

- With `--pretrain_from`, the program will only load the `state_dict` for the model, and write results to a new base directory.

## Evaluation

To **evaluate** a model, use `--test_only` and `--checkpoint` as follows

``` bash
python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --tag oliver \
    --test_only \
    --checkpoint <path-to-checkpoint> \
    DATASET.SPEAKER oliver
```

## Demo

To **evaluate** a model on an audio file, use `--demo_input` and `--checkpoint` as follows

```bash
python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --tag oliver \
    --demo_input demo_audio.wav \
    --checkpoint <path-to-checkpoint> \
    DATASET.SPEAKER oliver
```

You can find our checkpoint [here](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/qianshh_shanghaitech_edu_cn/EhOVnrnCYS5KqDIkamXBJbgBLOzu8vEFGwy88jSRSNATFA?e=Hc0cOO).

## FTD computation and template vector extraction
### Pose sequence reconstruction with VAE
First, you need to train the VAE by pose sequence reconstruction:

```bash
python main.py --config_file configs/pose2pose.yaml \
    --tag oliver \
    DATASET.SPEAKER oliver
```

### Compute FTD while training SDT-BP
Once the VAE is train, you can compute FTD while training our **SDT-BP** model by spotting out `VOICE2POSE.POSE_ENCODER.AE_CHECKPOINT` as follows:

```bash
python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --tag oliver \
    DATASET.SPEAKER oliver \
    VOICE2POSE.POSE_ENCODER.AE_CHECKPOINT <path-to-VAE-checkpoint>
```

### Training SDT-VAE
By changing the config file and spotting out `VOICE2POSE.POSE_ENCODER.AE_CHECKPOINT`, you can train our **SDT-VAE** model, and the FTD metric will also be computed:

```bash
python main.py --config_file configs/voice2pose_sdt_vae.yaml \
    --tag oliver \
    DATASET.SPEAKER oliver \
    VOICE2POSE.POSE_ENCODER.AE_CHECKPOINT <path-to-VAE-checkpoint>
```
For evaluation and demo with our SDT-VAE model, dont't forget to always specify the `VOICE2POSE.POSE_ENCODER.AE_CHECKPOINT` parameter.

## Misc

- We save a checkpoint and conduct validation after each epoch. You can change the interval in the config file.

- We generate and save 2 videos in each epoch when training. During validation, we sample 8 videos for each epoch. These videos can be saved in tensorborad (without sound) and mp4 (with sound). You can change the `SYS.VIDEO_FORMAT` parameter to select one or two of them.

- For multi-GPU training, we recommand using DistributedDataParallel (DDP) because it provide SyncBN across GPU cards. To enable DDP, set `SYS.DISTRIBUTED` to `True` and set `SYS.WORLD_SIZE` according to the number of GPUs.
    > When using DDP, assure that the `batch_size` can be divided exactly by `SYS.WORLD_SIZE`.

- We usually set `NUM_WORKERS` to 32 for best performance. If you encounter any error about memory, try lower `NUM_WORKERS`.


- We also support dataset caching (`DATASET.CACHING`) to further speed up data loading.
    > If you encounter errors in the dataloader like `RuntimeError: received 0 items of ancdata`, please increase `ulimit` by running the command `ulimit -n 262144`. (refer to this [issue](https://github.com/pytorch/pytorch/issues/973))

- To run any module other than the main files in the root directory, for example the `core\datasets\gesture_dataset.py` file, you should run `python -m core.datasets.gesture_dataset` rather than `python core\datasets\gesture_dataset.py`. This is an interesting problem of Python's relative importing.


```
@inproceedings{qian2021speech,
  title={Speech Drives Templates: Co-Speech Gesture Synthesis with Learned Templates},
  author={Qian, Shenhan and Tu, Zhi and Zhi, YiHao and Liu, Wen and Gao, Shenghua},
  journal={International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
