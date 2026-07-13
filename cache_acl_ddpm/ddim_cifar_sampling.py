import sys
sys.path.append("./mainldm")
sys.path.append("./mainddpm")
sys.path.append('./src/taming-transformers')
sys.path.append('.')
print(sys.path)
import argparse
import traceback
import shutil
import logging
import yaml
import random
import os, logging, gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
from tqdm import tqdm

from ddpm.utils.tools import set_random_seed
from accelerate import Accelerator, DistributedDataParallelKwargs
from ddpm.utils.utils import AttentionMap, AttentionMap_add, seed_everything, Fisher
from randomddpm.test.raw import split_zero_next_pairs

import matplotlib.pyplot as plt
torch.set_printoptions(sci_mode=False)
logger = logging.getLogger(__name__)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

"""
python randomddpm/ddim_cifar_sampling.py --random_cache --none_cache_position --best_population
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, default="./randomddpm/configs/cifar10.yml", help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234+9, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp", type=str, default="deepcache", help="Path for saving running related data.")
    parser.add_argument("--image_folder", type=str, default="./dataset/cifar10-sample-image", help="folder name for storing the sampled images")
    parser.add_argument("--fid", action="store_true", default=True)
    parser.add_argument("--interpolation", action="store_true", default=False)
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training")
    parser.add_argument("--ni", action="store_true", default=True, help="No interaction. Suitable for Slurm Job launcher",)
    parser.add_argument("--use_pretrained", action="store_true", default=True)
    parser.add_argument("--sample_type", type=str, default="generalized", help="sampling approach (generalized or ddpm_noisy)",)
    parser.add_argument("--skip_type", type=str, default="quad", help="skip according to (uniform or quadratic)",)
    parser.add_argument("--timesteps", type=int, default=100, help="number of steps involved")
    parser.add_argument("--eta", type=float, default=0.0, help="eta used to control the variances of sigma",)
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--select_step", type=int, default=None)
    parser.add_argument("--select_depth", type=int, default=None)
    parser.add_argument("--cache", action="store_true", default=True)
    parser.add_argument("--cache_interval", type=int, default=10,)
    parser.add_argument("--non_uniform", action="store_true", default=False)
    parser.add_argument("--pow", type=float, default=None,)
    parser.add_argument("--center", type=int, default=None,)
    parser.add_argument("--branch", type=int, default=2,)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--sample_batch', type=int, default=500)
    parser.add_argument("--dps_steps", action="store_true", default=False)
    parser.add_argument("--ptq", action="store_true", default=False)

    parser.add_argument("--random_cache", action="store_true", default=False, help="使用非默认缓存位置，默认生成随机缓存位置")
    parser.add_argument("--none_cache_position", action="store_true", default=False, help="使用预设的非单一缓存位置，即缓存序列")
    parser.add_argument("--best_population", action="store_true", default=False, help="使用最佳种群序列")
    parser.add_argument("--cache_position", type=int, default=None, choices=[1, 2, 3, 4])
    # 从环境变量或命令行参数获取budget
    parser.add_argument("--budget", type=int, default=25000, help="Budget value")
    args = parser.parse_args()
    if args.dps_steps:
        args.mode = "dps_opt"
    else:
        args.mode = "uni"
    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.select_step = args.select_step
    new_config.select_depth = args.select_depth
    torch.backends.cudnn.benchmark = True

    args, config = args, new_config
    accelerator = Accelerator()
    args.accelerator = accelerator
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("./run.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logging.info("start!")
    seed_everything(args.seed)
    # logging.info(args)

    """cifar10"""
    # interval_seq, all_cali_data, all_t, all_cali_t, all_cache \
    #     = torch.load("../CacheQuant/calibration/cifar{}_cache{}_{}.pth".format(args.timesteps, args.cache_interval, args.mode))
    # interval_seq = [0,5,13,20,24,37,48,83,84,94] #fid:4.9
    # interval_seq = [0, 10, 18, 21, 25, 34, 46, 65, 84, 92] #fid : 6.1260
    # interval_seq = [0, 5, 12, 19, 25, 32, 40, 47, 74, 81] # FID分数: 4.2686
    # interval_seq = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    # population_file =                                    |   种群pth文件
    # population_file_name                             |   种群pth文件名，不包含目录
    # population_file_result_folder                   |    结果保存文件加，文件名同种群pth文件名
    """读取种群保存的缓存全序列，然后生成缓存序列-fid这种数据格式的文件，保存"""
    # population_file =f"randomddpm/predictor/exps/cifar10_best_population/final_population_budget{args.budget}_seed3154.pth"
    # population_file_name = population_file.split('/')[-1].split('.')[0]
    # population_file_result_folder = result = '/'.join(population_file.split('/')[:-2] + [f"{population_file.split('/')[-2]}_pairs"]) + '/'

    population_file =f"randomddpm/predictor/exps/cifar_Ablation_Study_population/final_population_budget{args.budget}_seed3154_p80.pth"
    population_file_name = population_file.split('/')[-1].split('.')[0]
    population_file_result_folder = result = '/'.join(population_file.split('/')[:-2] + [f"{population_file.split('/')[-2]}_pairs"]) + '/'
    ms_seq = torch.load(population_file)[0]
    print(ms_seq)
    _,interval_seq,cache_seq= split_zero_next_pairs(ms_seq)
    logging.info(interval_seq)
    logging.info(cache_seq)

    args.interval_seq = interval_seq
    from ddpm.runners.deepcache import Diffusion
    runner = Diffusion(args, config, interval_seq=args.interval_seq)
    model = runner.creat_model()

    # del (all_cali_data, all_t, all_cali_t, all_cache)
    seed_everything(args.seed)
    # if self.args.random_cache:100个批次，每个批次500张图片，256channel_position，每个批次的缓存位置都是任意的，一共3^1000个方案
    # runner.sample_fid(model, total_n_samples=args.num_samples)
    # if self.args.random_cache:100个批次，每个批次500张图片，256channel_position，100个批次的缓存位置都是相同的，一共3^10个方案
    # runner.sample_fid_position_matrix_file(model,
    #                                   total_n_samples=args.num_samples,
    #                                   cache_sequence=cache_seq,
    #                                   full_cache_sequence = ms_seq,
    #                                   population_file_result_folder= population_file_result_folder,
    #                                   population_file_name = population_file_name)
    runner.sample_fid_position_matrix(model,
                                      total_n_samples=args.num_samples,
                                      cache_sequence=cache_seq,
                                      full_cache_sequence = ms_seq,
                                        )
    logging.info("sample cali finish!")



