import sys

sys.path.append("./randomldm")
sys.path.append("./randomddpm")
sys.path.append('./src/taming-transformers')
sys.path.append('.')
print(sys.path)

import argparse
import os, gc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import logging
import numpy as np
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from imwatermark import WatermarkEncoder
from PIL import Image
from einops import rearrange
import cv2
from tqdm import tqdm
from quant.utils import seed_everything
from randomddpm.test.raw import split_zero_next_pairs

logger = logging.getLogger(__name__)


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()

    """这里测量mac"""
    import sys
    sys.path.append('../')
    from ldm.flops import count_ops_and_params

    # 获取UNet
    unet = model.model.diffusion_model
    # MACs测量输入示例
    example_inputs = {
        'x': torch.randn(1, 3, 64, 64).cuda(),  # [1, 3, 64, 64]
        'timesteps': torch.ones(1).cuda(),  # [1]
        'context': torch.randn(1, 1, 512).cuda(),  # [1, 1, 512]
        # 'context':None,
        'y': None,  # 不使用y
        'prv_f': None,  # 无缓存
        # 'prv_f':torch.randn(1, 960, 8, 8).cuda()
    }

    macs, nparams = count_ops_and_params(unet, example_inputs=example_inputs, layer_wise=True)
    print("#Params: {:.4f} M".format(nparams / 1e6))
    print("#MACs: {:.4f} G".format(macs / 1e9))

    # self.logger.log("#Params: {:.4f} M".format(nparams / 1e6))
    # self.logger.log("#MACs: {:.4f} G".format(macs / 1e9))
    exit()
    return model


def get_model():
    config = OmegaConf.load("./randomldm/configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config(config, "../CacheQuant/models/ldm/cin256/model.ckpt")
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--sample_batch', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument('--ddim_steps', type=int, default=250)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1234+9)

    parser.add_argument("--replicate_interval", type=int, default=20)
    # parser.add_argument("--sm_abit",type=int, default=8)
    # parser.add_argument("--quant_act", action="store_true", default=True)
    # parser.add_argument("--weight_bit",type=int,default=8)
    # parser.add_argument("--act_bit",type=int,default=8)
    # parser.add_argument("--quant_mode", type=str, default="qdiff", choices=["qdiff"])
    # parser.add_argument("--lr_w",type=float,default=5e-3)
    # parser.add_argument("--lr_a", type=float, default=1e-4)
    # parser.add_argument("--lr_z",type=float,default=1e-1)
    # parser.add_argument("--lr_rw",type=float,default=1e-2)
    # parser.add_argument("--split", action="store_true", default=True)
    parser.add_argument("--ptq", action="store_true", default=False)
    parser.add_argument("--dps_steps", action='store_true', default=True)

    parser.add_argument("--nonuniform", action='store_true', default=False)
    parser.add_argument("--pow", type=float, default=1.5)

    parser.add_argument("--imglogdir", type=str, default="./dataset/imagenet-sample-image",help="生成的图像保存目录")
    parser.add_argument("--random_cache", action="store_true", default=False, help="使用非默认缓存位置，默认生成随机缓存位置")
    parser.add_argument("--none_cache_position", action="store_true", default=None, help="使用预设的非单一缓存位置，即自定义缓存序列")
    parser.add_argument("--cache_position", type=int, default=None, choices=[1, 2, 3, 4],help="使用预设的单一缓存位置")

    # 从环境变量或命令行参数获取budget
    parser.add_argument("--budget", type=int, default=12500, help="Budget value")
    args = parser.parse_args()

    if args.dps_steps:
        args.mode = "dps_opt"
    else:
        args.mode = "uni"

    seed_everything(args.seed)
    device = torch.device("cuda", args.local_rank)

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
    logging.info(args)
    seed_everything(args.seed)
    """这里预留我们读取种群文件的逻辑"""
    # # population_file =                                    |   种群pth文件
    # # population_file_name                             |   种群pth文件名，不包含目录
    # # population_file_result_folder                   |    结果保存文件加，文件名同种群pth文件名

    population_file =f"randomldm/predictor/exps/imagenet_best_population/final_population_budget{args.budget}_seed6688.pth"
    population_file_name = population_file.split('/')[-1].split('.')[0]
    population_file_result_folder = result = '/'.join(population_file.split('/')[:-2] + [f"{population_file.split('/')[-2]}_pairs"]) + '/'

    ms_sequence = torch.load(population_file)[0]
    logging.info(ms_sequence)
    _,interval_seq,cache_sequence= split_zero_next_pairs(ms_sequence)
    logging.info(interval_seq)
    logging.info(cache_sequence)

    # 只加载interval_seq，不需要其他校准数据 ddim_sampling中加载非均匀缓存间隔,下面保存的数据是DPS得到的时间步 interval = 10 /20
    # interval_seq = [0, 40, 73, 93, 110, 126, 142, 159, 176, 193, 210, 226, 240]# 不使用这里的seq，使用后面函数生成的seq
    # interval_seq = [0, 20, 40, 56, 68, 77, 86, 95, 104, 113, 122, 130, 139, 148, 157, 166, 175, 185, 195, 205, 214, 223, 231, 239, 245]
    # interval_seq = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
    # interval_seq = None
    args.interval_seq = interval_seq
    logger.info(f"The interval_seq: {args.interval_seq}")

    # 加载模型
    model = get_model()

    # 不加载a_list/b_list
    model.model.diffusion_model.timesteps = args.ddim_steps

    # 创建采样器（无量化）
    sampler = DDIMSampler(model, slow_steps=args.interval_seq)
    # ms_sequence = None
    # cache_sequence = None
    ms_seq =ms_sequence
    cache_seq = cache_sequence

    model.model.reset_no_cache(no_cache=False)
    model.model.diffusion_model.time = 0

    # 设置输出目录
    imglogdir = args.imglogdir
    os.makedirs(imglogdir, exist_ok=True)
    base_count = 0

    # 水印设置
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    logging.info("sampling...")
    num_batches = args.num_samples // args.sample_batch  # 2000个批次

    # 记录开始时间
    start_time = time.time()
    iterator = tqdm(range(num_batches), desc='DDIM Sampler')

    # iterator = tqdm(range(1000), desc='DDIM Sampler')
    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(args.sample_batch * [1000]).to(model.device)}
            )

            for i, class_num in enumerate(iterator):
                class_label = class_num // 2
                print("当前类别:",class_label)
                # class_label = class_num % args.num_classes
                # class_label = class_num
                xc = torch.tensor(args.sample_batch * [class_label])

                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

                # 采样一个batch，也就是25张图片， 这是在潜空间表示的，里面做250步的ddim，args.ddim_steps
                # samples_ddim, _ = sampler.sample(
                #     S=args.ddim_steps,
                #     conditioning=c,
                #     batch_size=args.sample_batch,
                #     shape=[3, 64, 64],
                #     verbose=False,
                #     unconditional_guidance_scale=args.scale,
                #     unconditional_conditioning=uc,
                #     eta=args.ddim_eta,
                #     replicate_interval=args.replicate_interval,
                #     nonuniform=args.nonuniform,
                #     pow=args.pow
                # )

                samples_ddim, _ = sampler.sample_position_matrix(
                    S=args.ddim_steps,
                    conditioning=c,
                    batch_size=args.sample_batch,
                    shape=[3, 64, 64],
                    verbose=False,
                    unconditional_guidance_scale=args.scale,
                    unconditional_conditioning=uc,
                    eta=args.ddim_eta,
                    replicate_interval=args.replicate_interval,
                    nonuniform=args.nonuniform,
                    pow=args.pow,
                    cache_sequence = cache_seq,
                    full_cache_sequence = ms_seq,
                    args = args
                )

                # 解码生成图像 这是在像素空间表示的
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                x_checked_image = x_samples_ddim
                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                # 保存图像
                for x_sample in x_checked_image_torch:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(imglogdir, f"{base_count:05}.png"))
                    base_count += 1

                    if base_count == args.num_samples:
                        break

                if base_count == args.num_samples:
                    break

                # exit() #测量latency

    # 计算总时间和吞吐量
    total_time = time.time() - start_time
    throughput = base_count / total_time

    logger.info("=" * 50)
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Throughput: {throughput:.2f} images/second")
    logger.info("=" * 50)

    # 打印ms_seq信息
    logger.info("=" * 50)
    logger.info(f"  - Model Schedule length: {len(ms_seq)}")
    logger.info("=" * 50)
    logging.info("sample finish!")