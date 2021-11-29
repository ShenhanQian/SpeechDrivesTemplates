import os
import warnings
warnings.simplefilter("ignore")
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from configs.default import get_cfg_defaults
from core.pipelines import get_pipeline


def setup_config():
    parser = argparse.ArgumentParser(description="voice2pose main program")
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--resume_from", type=str, default=None, help="the checkpoint to resume from")
    parser.add_argument("--test_only", action="store_true", help="perform testing and evaluation only")
    parser.add_argument("--demo_input", type=str, default=None, help="path to input for demo")
    parser.add_argument("--checkpoint", type=str, default=None, help="the checkpoint to test with")
    parser.add_argument("--tag", type=str, default='', help="tag for the experiment")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return args, cfg

def run(args, cfg):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    pipeline = get_pipeline(cfg.PIPELINE_TYPE)(cfg)

    cfg_name = args.config_file.split('/')[-1].split('.')[0]
    if args.demo_input:
        exp_tag = cfg_name + '-DEMO-' + args.tag
        pipeline.demo(cfg, exp_tag, args.checkpoint, args.demo_input)
    elif args.test_only:
        exp_tag = cfg_name + '-TEST-' + args.tag
        pipeline.test(cfg, exp_tag, args.checkpoint)
    else:
        exp_tag = cfg_name + '-TRAIN-' + args.tag
        pipeline.train(cfg, exp_tag, args.resume_from)

def run_distributed(rank, args, cfg):
    os.environ['MASTER_ADDR'] = cfg.SYS.MASTER_ADDR
    os.environ['MASTER_PORT'] = str(cfg.SYS.MASTER_PORT)
    dist.init_process_group("nccl", rank=rank, world_size=cfg.SYS.WORLD_SIZE)

    run(args, cfg)

def main():
    args, cfg = setup_config()

    if cfg.SYS.DISTRIBUTED:
        mp.spawn(run_distributed, 
            args=(args, cfg),
            nprocs=cfg.SYS.WORLD_SIZE,
            join=True)
    else:
        run(args, cfg)  
    

if __name__ == "__main__":
    main()
