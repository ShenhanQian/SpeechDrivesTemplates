# SpeechDrivesTemplates
The official repo for the ICCV-2021 paper "Speech Drives Templates: Co-Speech Gesture Synthesis with Learned Templates".

[[arxiv](https://arxiv.org/abs/2108.08020) / [video](https://youtu.be/yu-5gUHn6h8)]

<p align="center">
  <img src="./iccv2021_sdt.jpg" width=500px/>
</p>

- [X] Code
- [ ] Model
- [ ] Data preparation


## Package Hierarchy

```
|-- config
|     |-- default.py
|     |-- voice2pose_s2g_speech2gesture.yaml        # baseline: speech2gesture
|     |-- voice2pose_sdt_vae_speech2gesture.yaml    # ours (VAE)
|     |-- pose2pose_speech2gesture.yaml             # gesture reconstruction  
|     `-- voice2pose_sdt_bp_speech2gesture.yaml     # ours (Backprop)
|
|-- core
|     |-- datasets
|     |-- netowrks
|     |-- pipelines
|     \-- utils
|
|-- dataset
|     \-- speech2gesture  # create a soft link here
|
|-- output
|     \-- <date-config-tag>  # A directory for each experiment
|
`-- main.py

```

We split the entire pipeline into two sub-pipelines including
`voice2pose` and `pose2pose`, which inherit from the `Trainer` class.

## Setup the Dataset

Datasets shuold be placed in the `dataset` directory. Just create a soft link like this:

``` bash
ln -s <path-to-SPEECH2GESTURE-dataset> ./dataset/speech2gesture
```

For your own dataset, you need to implement a subclass of `torch.utils.data.Dataset` in `core/datasets/custom_dataset.py`.

## Train

### Train a Model from Scratch

``` bash
python main.py --config_file configs/voice2pose_sdt_bp_speech2gesture.yaml \
    --tag DEV \
    SYS.NUM_WORKERS 32
```

- `--tag` set the name of the experiment which wil be displayed in the outputfile.
- You can overwrite the any parameters defined in `voice2pose_default.py` by simply
adding it at the end of the command. The example above set `SYS.NUM_WORKERS` to 32 temporarily.

### Resume Training from an Interrupted Experiment

``` bash
python main.py --config_file configs/voice2pose_sdt_bp_speech2gesture.yaml \
    --resume_from <checkpoint-to-continue-from>
```

- This command will load the `state_dict` from the checkpoint for both the model and the optimizer, and write results to the original directory that the checkpoint lies in.

### Training from a pretrained model

``` bash
python main.py --config_file configs/voice2pose_sdt_bp_speech2gesture.yaml \
    --pretrain_from <checkpoint-to-continue-from> \
    --tag DEV
```

- This command will only load the `state_dict` for the model, and write results to a new base directory.

## Test

To **test** the model, run this command:

``` bash
python main.py --config_file configs/voice2pose_sdt_bp_speech2gesture.yaml \
    --tag DEV \
    --test-only \
    --checkpoint <path-to-checkpoint>
```

## Demo

``` bash
python main.py --config_file configs/voice2pose_sdt_bp_speech2gesture.yaml \
    --tag <DEV> \
    --demo_input <audio.wav> \
    --checkpoint <path-to-checkpoint> \
    DATASET.SPEAKER oliver \
    SYS.VIDEO_FORMAT "['mp4']"
```

## Important Details
### Dataset caching
We turn on dataset caching (`DATASET.CACHING`) by default to speed up training. 

> If you encounter errors in the dataloader like `RuntimeError: received 0 items of ancdata`, please increase `ulimit` by running the command `ulimit -n 262144`. (refer to this [issue](https://github.com/pytorch/pytorch/issues/973))
### DataParallel and DistributedDataParallel
We use single GPU (warpped by DataParallel) by default since it is fast enough with dataset caching. For multi-GPU training, we recommand using DistributedDataParallel (DDP) because it provide SyncBN across GPU cards. To enable DDP, set `SYS.DISTRIBUTED` to `True` and set `SYS.WORLD_SIZE` according to the number of GPUs.
> When using DDP, assure that the `batch_size` can be divided exactly by `SYS.WORLD_SIZE`.

## Misc
- To run any module other than the main files in the root directory, for example the `core\datasets\speech2gesture.py` file, you should run `python -m core.datasets.speech2gesture` rather than `python core\datasets\speech2gesture.py`. This is an interesting problem of Python's relative importing which deserves in-depth thinking.
- We save a checkpoint and conduct validation after each epoch. You can change the interval in the config file.
- We generate and save 2 videos in each epoch when training. During validation, we sample 8 videos for each epoch. These videos are saved in tensorborad (without sound) and mp4 (with sound). You can change the `SYS.VIDEO_FORMAT` parameter to select one or two of them.
- We usually sett `NUM_WORKERS` to 32 for best performance. If you encounter any error about memory, try lower `NUM_WORKERS`.


```
@inproceedings{qian2021speech,
  title={Speech Drives Templates: Co-Speech Gesture Synthesis with Learned Templates},
  author={Qian, Shenhan and Tu, Zhi and Zhi, YiHao and Liu, Wen and Gao, Shenghua},
  journal={International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
