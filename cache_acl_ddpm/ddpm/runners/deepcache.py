import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch.nn.functional import adaptive_avg_pool2d
import shutil
import  random
from ..models.ema import EMAHelper
from ..functions import get_optimizer
from ..functions.losses import loss_registry
from ..datasets import get_dataset, data_transform, inverse_data_transform
from ..functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
from ..utils import tools
from ..utils.cache_schedule_util import get_cache_schedule
logger = logging.getLogger(__name__)

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, interval_seq=None):
        self.args = args
        self.config = config
        self.accelerator = args.accelerator
        self.device = self.accelerator.device
        self.config.device = self.device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.interval_seq = interval_seq

# 原始创建模型方法
    # def creat_model(self):
    #     if self.args.cache:
    #         from ..models.deepcache_diffusion import Model
    #         model = Model(self.config)
    #         model.set_cache_para(self.args.branch) # 这里就是根据branch，设置缓存的层
    #         logger.info('Sampling in DeepCache mode')
    #     else:
    #         raise ValueError
    #
    #     if self.config.data.dataset == "CIFAR10":
    #         name = "cifar10"
    #     elif self.config.data.dataset == "LSUN":
    #         name = f"lsun_{self.config.data.category}"
    #     else:
    #         raise ValueError
    #     ckpt = get_ckpt_path(f"ema_{name}")
    #     logger.info("Loading checkpoint {}".format(ckpt))
    #     msg = model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)
    #
    #     logger.info(msg)
    #     model.cuda()
    #     model.eval()
    #     return model

    def creat_model(self, dataset_generate_mode=False):
        if self.args.cache:
            from ..models.deepcache_diffusion import Model
            model = Model(self.config)

            # 如果是数据集生成模式，设置特殊标记
            if dataset_generate_mode:
                # 数据集生成模式：模型将使用外部传入的缓存序列
                model.set_dataset_generate_mode()
                logger.info('模型设置为数据集生成模式，将使用外部缓存序列')

            else:
                # 常规采样模式：使用原有的逻辑
                if self.args.random_cache:
                    if hasattr(self.args, 'cache_position') and self.args.cache_position is not None:
                        model.set_random_cache_para(cache_index=self.args.cache_position-1)
                        logger.info(f'使用指定缓存位置: branch={self.args.cache_position}')
                    elif hasattr(self.args, 'best_population'):

                        logger.info('使用最佳种群的随机缓存位置')
                    else:
                        model.set_random_cache_para()
                        logger.info('使用随机缓存位置')
                else:
                    model.set_cache_para(self.args.branch)
                    logger.info(f'使用固定缓存位置: branch={self.args.branch}')

            logger.info('Sampling in DeepCache mode')
        else:
            raise ValueError("DeepCache mode is required for this experiment")

        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError

        ckpt = get_ckpt_path(f"ema_{name}")
        logger.info("Loading checkpoint {}".format(ckpt))
        msg = model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)

        logger.info(msg)
        model.cuda()
        model.eval()

        # import sys
        # sys.path.append('../')
        # from ..utils.flops import count_ops_and_params
        # example_inputs = {
        #     'x': torch.randn(1, 3, self.config.data.image_size, self.config.data.image_size).to(self.device),
        #     't': torch.ones(1).to(self.device),
        #     'prv_f': torch.randn(1, 128, 128, 128).to(self.device),
        #     # 'prv_f': None
        # }
        # macs, nparams = count_ops_and_params(model, example_inputs=example_inputs, layer_wise=True)
        # print("#Params: {:.4f} M".format(nparams / 1e6))
        # print("#MACs: {:.4f} G".format(macs / 1e9))
        # # self.logger.log("#Params: {:.4f} M".format(nparams / 1e6))
        # # self.logger.log("#MACs: {:.4f} G".format(macs / 1e9))
        # exit()

        return model

    def sample(self):
        if self.args.cache:
            from ..models.deepcache_diffusion import Model
            model = Model(self.config)
            logger.info('Sampling in DeepCache mode')
        else:
            from ..models.diffusion import Model
            model = Model(self.config)
       
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.logger.checkpoint_path, "ckpt.pth"),
                    map_location='cpu',
                )
                logger.info("Loading from latest checkpoint: {}".format(
                    os.path.join(self.logger.checkpoint_path, "ckpt.pth")
                ))
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location='cpu',
                )
                logger.info("Loading from latest checkpoint: {}".format(
                    os.path.join(self.logger.checkpoint_path, f"ckpt_{self.config.sampling.ckpt_id}.pth")
                ))
            model.load_state_dict(tools.unwrap_module(states[0]), strict=True)
            
            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(tools.unwrap_module(states[-1]))
                ema_helper.ema(model)
            else:
                ema_helper = None
            
            model = self.accelerator.prepare(model)
        else:
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            logger.info("Loading checkpoint {}".format(ckpt))
            msg = model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)

            logger.info(msg)
            model = self.accelerator.prepare(model)

        model.eval()
        import sys
        sys.path.append('../')
        from flops import count_ops_and_params
        example_inputs = {
            'x': torch.randn(1, 3, self.config.data.image_size, self.config.data.image_size).to(self.device),
            't': torch.ones(1).to(self.device),
            'prv_f': [torch.randn(1, 256, 16, 16).to(self.device)],
            'branch': 2
        }
        macs, nparams = count_ops_and_params(model, example_inputs=example_inputs, layer_wise=False)
        self.logger.log("#Params: {:.4f} M".format(nparams/1e6))
        self.logger.log("#MACs: {:.4f} G".format(macs/1e9))
        exit()

        if self.args.fid:
            self.sample_fid(model, total_n_samples=self.args.max_images)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    # 原始的sample_fid
    # 为什么我们要改进这里呢，不直接使用随机缓存的，我们的目的是为了让100个批次生成的数据，利用的缓存点一样，批次内相同！不然有4^1000种方案
    def sample_fid(self, model, total_n_samples=50000, save_images = True, timesteps=None):


        config = self.config
        # img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        img_id = 0
        logger.info(f"starting from image {img_id}")
        total_n_samples = total_n_samples // self.accelerator.num_processes
        # n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        n_rounds = (total_n_samples - img_id) // self.args.sample_batch

        generate_samples = []
        throughput = []
        sample_start_time = time.time()
        with torch.no_grad(), tqdm.tqdm(range(n_rounds)) as t:
            for _ in t:
                start_time = time.time()
                # n = config.sampling.batch_size
                n = self.args.sample_batch
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model, timesteps=timesteps)
                x = inverse_data_transform(config, x)

                use_time = time.time() - start_time
                throughput.append(x.shape[0] / use_time)
                t.set_description(f"Throughput: {np.mean(throughput):.2f} samples/s")
                
                if save_images:
                    for i in range(n):
                        tvu.save_image(
                            x[i], os.path.join(self.args.image_folder, f"{self.accelerator.process_index}_{img_id}.png")
                        )
                        
                        img_id += 1
                else:
                    generate_samples.append(x)
        
        self.args.accelerator.wait_for_everyone()
        logger.info(f"Time taken: {time.time() - sample_start_time} seconds")
        return generate_samples

    def sample_image(self, x, model, last=True, timesteps=None):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        if timesteps is None:
            timesteps = self.args.timesteps # 这里我们拿到的args.timesteps=100
        # print(self.args.sample_type, self.args.skip_type, timesteps)
            # 默认args.sample_type == "generalized"
        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            if self.interval_seq == None:
                from ..functions.deepcache_denoising import generalized_steps
                xs = generalized_steps(
                    x, seq, model, self.betas,
                    timesteps=timesteps,
                    cache_interval=self.args.cache_interval,  # for uniform
                    non_uniform=self.args.non_uniform, pow = self.args.pow, center = self.args.center, branch=self.args.branch,  # for non-uniform
                    eta=self.args.eta)
            else:
                # 这里使用DPS算法的时间步
                from ..functions.deepcache_denoising import adaptive_generalized_steps
                xs = adaptive_generalized_steps(
                    x, seq, model, self.betas,
                    timesteps=timesteps,
                    interval_seq = self.interval_seq, branch=self.args.branch,  # for non-uniform
                    eta=self.args.eta,
                    quant=self.args.ptq,
                    args=self.args)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            # Not implemented for DeepCache
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from ..functions.deepcache_denoising import ddpm_steps
            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x
    # """3block"""
    # def sample_fid_position_matrix(self, model, total_n_samples=50000, save_images=True, timesteps=None):
    #     config = self.config
    #     img_id = 0
    #     logger.info(f"starting from image {img_id}")
    #     total_n_samples = total_n_samples // self.accelerator.num_processes
    #     n_rounds = (total_n_samples - img_id) // self.args.sample_batch
    #
    #     # === 新增：预生成缓存位置序列 ===
    #     if self.args.random_cache:
    #         self.cache_sequence = self._generate_cache_sequence(model, len(self.interval_seq))
    #         # 生成描述用于日志
    #
    #         position_descriptions = {
    #             0: "up2.block2",
    #             1: "up1.block0",
    #             2: "up1.block1"
    #         }
    #         cache_descriptions = [position_descriptions[idx] for idx in self.cache_sequence]
    #         logger.info(f"所有批次使用缓存序列: {cache_descriptions}")
    #         # === 保存缓存位置到日志文件 ===
    #         self._save_cache_positions_to_log(self.cache_sequence)  # 传递 self.cache_sequence
    #     else:
    #         self.cache_sequence = None
    #
    #     generate_samples = []
    #     throughput = []
    #     sample_start_time = time.time()
    #     with torch.no_grad(), tqdm.tqdm(range(n_rounds)) as t:
    #         for _ in t:
    #             start_time = time.time()
    #             n = self.args.sample_batch
    #             x = torch.randn(
    #                 n,
    #                 config.data.channels,
    #                 config.data.image_size,
    #                 config.data.image_size,
    #                 device=self.device,
    #             )
    #
    #             # === 修改：传递缓存序列 ===
    #             x = self.sample_image_position_matrix(x, model, timesteps=timesteps, cache_sequence=self.cache_sequence)
    #             x = inverse_data_transform(config, x)
    #
    #             use_time = time.time() - start_time
    #             throughput.append(x.shape[0] / use_time)
    #             t.set_description(f"Throughput: {np.mean(throughput):.2f} samples/s")
    #
    #             if save_images:
    #                 for i in range(n):
    #                     tvu.save_image(
    #                         x[i], os.path.join(self.args.image_folder, f"{self.accelerator.process_index}_{img_id}.png")
    #                     )
    #                     img_id += 1
    #             else:
    #                 generate_samples.append(x)
    #
    #     self.args.accelerator.wait_for_everyone()
    #     logger.info(f"Time taken: {time.time() - sample_start_time} seconds")
    #     return generate_samples
    #
    # def _save_cache_positions_to_log(self, cache_indices):
    #     """保存缓存位置到日志文件"""
    #     log_file = "random_cache_position_fid_score.log"
    #
    #     # 位置描述映射
    #     position_descriptions = {
    #         0: "up2.block2",
    #         1: "up1.block0",
    #         2: "up1.block1"
    #     }
    #
    #     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #
    #     # 更清晰的格式：每个位置单独一行
    #     log_content = f"[{current_time}] 缓存位置序列:\n"
    #     for i, idx in enumerate(cache_indices):
    #         desc = position_descriptions[idx]
    #         log_content += f"  缓存点{i:2d}: {idx}-{desc}\n"
    #     log_content += "\n"
    #
    #     try:
    #         with open(log_file, "a", encoding="utf-8") as f:
    #             f.write(log_content)
    #         logger.info(f"缓存位置已保存到: {log_file}")
    #     except Exception as e:
    #         logger.error(f"保存缓存位置日志失败: {e}")
    #
    # def _generate_cache_sequence(self, model, num_positions):
    #     """生成缓存位置序列（返回索引列表）"""
    #     import random
    #
    #     # 使用固定的缓存种子
    #     cache_seed = int(time.time() * 1000) % 1000000
    #     random.seed(cache_seed)
    #     logger.info(f"使用缓存种子: {cache_seed}")
    #
    #     # 生成索引序列 [0, 1, 2] 的随机组合
    #     sequence = [random.randint(0, 2) for _ in range(num_positions)]
    #
    #     # 位置描述映射
    #     position_descriptions = {
    #         0: "up2.block2",
    #         1: "up1.block0",
    #         2: "up1.block1"
    #     }
    #
    #     # descriptions = [position_descriptions[idx] for idx in sequence]
    #     # logger.info(f"生成的缓存序列: {descriptions}")
    #
    #     return sequence  # 返回索引列表 [0, 1, 2, 0, ...]

# 我真下头

    """
    4resolution
    逻辑是 首先记录实际位置到random_cache_position_fid_score.log ， 然后在计算fid脚本的时候，在脚本里面设置 log的位置是random_cache_position_fid_score.log
    并且记录fid指标，这样就有 位置-fid的数据
    """
    def sample_fid_position_matrix_file(self, model, total_n_samples=50000, cache_sequence = None,full_cache_sequence = None,population_file_result_folder= None,population_file_name= None ,save_images=True, timesteps=None):
        config = self.config
        img_id = 0
        logger.info(f"starting from image {img_id}")
        total_n_samples = total_n_samples // self.accelerator.num_processes
        n_rounds = (total_n_samples - img_id) // self.args.sample_batch

        # === 预生成缓存位置序列 ===
        if self.args.random_cache:
            # 判断是否使用预设位置
            use_preset = hasattr(self.args, 'cache_position') and self.args.cache_position is not None

            # 修改：根据条件选择缓存生成方式
            if use_preset:
                # 使用预设位置
                cache_position = self.args.cache_position
                self.full_cache_sequence, self.used_cache_sequence, self.cache_sequence = self._generate_cache_sequence(
                    model, len(self.interval_seq), fixed_position=cache_position
                )
            elif hasattr(self.args, 'none_cache_position'):
                self.cache_sequence = cache_sequence
                self.full_cache_sequence = full_cache_sequence
                self.used_cache_sequence = [x + 1 for x in self.cache_sequence]
            else:
                # 使用随机位置
                self.full_cache_sequence, self.used_cache_sequence, self.cache_sequence = self._generate_cache_sequence(
                    model, len(self.interval_seq)
                )
                    # self.cache_sequence = [3,0,0,0,0,0,0,3,3,3] # fid : 4.9
                    # self.cache_sequence = [3, 0, 0, 0, 0, 3, 0, 2, 3, 3] #fid : 6.1260
                    # self.cache_sequence = [3, 0, 0, 0, 0, 0, 3, 0, 3, 2] # FID分数: 4.2686
                    # """to do , 利用args.none_cache_position将这里的逻辑分开"""

            # 生成描述用于日志 - 更新为1-4的编码
            # position_descriptions = {
            #     0: "使用缓存",
            #     1: "up3.block1",
            #     2: "up2.block1",
            #     3: "up1.block1",
            #     4: "up0.block1"
            # }
            # cache_descriptions = [position_descriptions[idx] for idx in self.used_cache_sequence]

            # === 保存缓存位置到日志文件 ===
            self._save_cache_positions_to_log(self.used_cache_sequence)  # 传递实际使用的位置
        else:
            self.full_cache_sequence = None
            self.used_cache_sequence = None

        generate_samples = []
        throughput = []
        sample_start_time = time.time()
        with torch.no_grad(), tqdm.tqdm(range(n_rounds)) as t:
            for _ in t:
                start_time = time.time()
                n = self.args.sample_batch
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                # === 传递缓存序列 ===
                x = self.sample_image_position_matrix(x, model, timesteps=timesteps,
                                                      cache_sequence=self.cache_sequence)
                x = inverse_data_transform(config, x)

                use_time = time.time() - start_time
                throughput.append(x.shape[0] / use_time)
                t.set_description(f"Throughput: {np.mean(throughput):.2f} samples/s")

                if save_images:
                    for i in range(n):
                        tvu.save_image(
                            x[i], os.path.join(self.args.image_folder, f"{self.accelerator.process_index}_{img_id}.png")
                        )
                        img_id += 1
                else:
                    generate_samples.append(x)

        self.args.accelerator.wait_for_everyone()
        total_time = time.time() - sample_start_time
        logger.info(f"Time taken: {total_time} seconds")

        # === 新增：调用外部FID计算并保存到数据集 ===
        try:
            # 调用外部FID计算命令
            fid_score = self._calculate_external_fid()

            # 保存到数据集 - 使用完整的100时间步序列
            # self._save_to_dataset(self.full_cache_sequence, fid_score,population_file_name = population_file_name)
            self._save_to_dataset_population(self.full_cache_sequence, fid_score, population_file_result_folder=population_file_result_folder,population_file_name = population_file_name)
            logger.info(f"✅ 完成! FID: {fid_score:.4f}")

        except Exception as e:
            logger.error(f"❌ FID计算或数据保存失败: {e}")

        return generate_samples

    def sample_fid_position_matrix(self, model, total_n_samples=50000, cache_sequence = None,full_cache_sequence = None, save_images=True, timesteps=None):
        config = self.config
        img_id = 0
        logger.info(f"starting from image {img_id}")
        total_n_samples = total_n_samples // self.accelerator.num_processes
        n_rounds = (total_n_samples - img_id) // self.args.sample_batch

        # === 预生成缓存位置序列 ===
        if self.args.random_cache:
            # 判断是否使用预设位置
            use_preset = hasattr(self.args, 'cache_position') and self.args.cache_position is not None

            # 修改：根据条件选择缓存生成方式
            if use_preset:
                # 使用预设位置
                cache_position = self.args.cache_position
                self.full_cache_sequence, self.used_cache_sequence, self.cache_sequence = self._generate_cache_sequence(
                    model, len(self.interval_seq), fixed_position=cache_position
                )
            elif hasattr(self.args, 'none_cache_position'):
                self.cache_sequence = cache_sequence
                self.full_cache_sequence = full_cache_sequence
                self.used_cache_sequence = [x + 1 for x in self.cache_sequence]
            else:
                # 使用随机位置
                self.full_cache_sequence, self.used_cache_sequence, self.cache_sequence = self._generate_cache_sequence(
                    model, len(self.interval_seq)
                )
                    # self.cache_sequence = [3,0,0,0,0,0,0,3,3,3] # fid : 4.9
                    # self.cache_sequence = [3, 0, 0, 0, 0, 3, 0, 2, 3, 3] #fid : 6.1260
                    # self.cache_sequence = [3, 0, 0, 0, 0, 0, 3, 0, 3, 2] # FID分数: 4.2686
                    # """to do , 利用args.none_cache_position将这里的逻辑分开"""

            # 生成描述用于日志 - 更新为1-4的编码
            # position_descriptions = {
            #     0: "使用缓存",
            #     1: "up3.block1",
            #     2: "up2.block1",
            #     3: "up1.block1",
            #     4: "up0.block1"
            # }
            # cache_descriptions = [position_descriptions[idx] for idx in self.used_cache_sequence]

            # === 保存缓存位置到日志文件 ===
            self._save_cache_positions_to_log(self.used_cache_sequence)  # 传递实际使用的位置
        else:
            self.full_cache_sequence = None
            self.used_cache_sequence = None

        generate_samples = []
        throughput = []
        sample_start_time = time.time()
        with torch.no_grad(), tqdm.tqdm(range(n_rounds)) as t:
            for _ in t:
                start_time = time.time()
                n = self.args.sample_batch
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                # === 传递缓存序列 ===
                x = self.sample_image_position_matrix(x, model, timesteps=timesteps,
                                                      cache_sequence=self.cache_sequence)
                x = inverse_data_transform(config, x)

                use_time = time.time() - start_time
                throughput.append(x.shape[0] / use_time)
                t.set_description(f"Throughput: {np.mean(throughput):.2f} samples/s")

                if save_images:
                    for i in range(n):
                        tvu.save_image(
                            x[i], os.path.join(self.args.image_folder, f"{self.accelerator.process_index}_{img_id}.png")
                        )
                        img_id += 1
                else:
                    generate_samples.append(x)

        self.args.accelerator.wait_for_everyone()
        total_time = time.time() - sample_start_time
        logger.info(f"Time taken: {total_time} seconds")
        self._save_time_taken_to_log(total_time)  # 传递实际使用的时间
        # === 新增：调用外部FID计算并保存到数据集 ===
        try:
            # 调用外部FID计算命令
            fid_score = self._calculate_external_fid()

            # 保存到数据集 - 使用完整的100时间步序列
            # self._save_to_dataset(self.full_cache_sequence, fid_score)
            # self._save_to_dataset_population(self.full_cache_sequence, fid_score, population_file_result_folder=population_file_result_folder,population_file_name = population_file_name)
            logger.info(f"✅ 完成! FID: {fid_score:.4f}")

        except Exception as e:
            logger.error(f"❌ FID计算或数据保存失败: {e}")

        return generate_samples

    def sample_image_position_matrix(self, x, model, last=True, timesteps=None, cache_sequence=None):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        if timesteps is None:
            timesteps = self.args.timesteps # 这里我们拿到的args.timesteps=100
        # print(self.args.sample_type, self.args.skip_type, timesteps)
            # 默认args.sample_type == "generalized"
        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            if self.interval_seq == None:
                from ..functions.deepcache_denoising import generalized_steps
                xs = generalized_steps(
                    x, seq, model, self.betas,
                    timesteps=timesteps,
                    cache_interval=self.args.cache_interval,  # for uniform
                    non_uniform=self.args.non_uniform, pow = self.args.pow, center = self.args.center, branch=self.args.branch,  # for non-uniform
                    eta=self.args.eta)
            else:
                # 这里使用DPS算法的时间步 /自定义时间步
                from ..functions.deepcache_denoising import adaptive_generalized_steps
                xs = adaptive_generalized_steps(
                    x, seq, model, self.betas,
                    timesteps=timesteps,
                    interval_seq = self.interval_seq, branch=self.args.branch,  # for non-uniform
                    eta=self.args.eta,
                    quant=self.args.ptq,
                    args=self.args,
                    cache_sequence=cache_sequence  )# 新增参数
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            # Not implemented for DeepCache
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from ..functions.deepcache_denoising import ddpm_steps
            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def _generate_cache_sequence(self, model, num_positions, fixed_position=None):
        """根据interval_seq生成100个时间步的序列"""
        import random

        if fixed_position is not None:
            # 使用固定位置模式
            logger.info(f"使用固定缓存位置: {fixed_position}")

            # 初始化100个时间步，全部设为0（使用缓存）
            full_sequence = [0] * 100

            # 生成固定位置的序列（所有位置都使用相同的固定值）
            sequence = [fixed_position - 1 for _ in range(num_positions)]  # 1-4 变成 0-3

            # 在interval_seq指定的索引位置设置缓存位置
            for i, idx in enumerate(self.interval_seq):
                if idx < len(full_sequence) and i < len(sequence):  # 确保索引在范围内
                    cache_position = sequence[i] + 1  # 0-3 变成 1-4
                    full_sequence[idx] = cache_position

            # 实际使用的位置就是固定位置
            used_positions = [fixed_position] * num_positions

            logger.info(f"interval_seq: {self.interval_seq}")
            logger.info(f"生成的固定缓存序列: {sequence}")
            logger.info(f"生成的100时间步序列: {full_sequence}")
            logger.info(f"实际使用的缓存位置: {used_positions}")

            # 返回三个值：完整序列、实际使用的位置和原始序列
            return full_sequence, used_positions, sequence

        else:
            # 原有的随机生成逻辑
            # 初始化100个时间步，全部设为0（使用缓存）
            full_sequence = [0] * 100

            # 生成索引序列 [0, 1, 2, 3] 的随机组合（四个位置）
            sequence = [random.randint(0, 3) for _ in range(num_positions)]

            # 在interval_seq指定的索引位置设置缓存位置，使用sequence中的值+1（变成1-4）
            for i, idx in enumerate(self.interval_seq):
                if idx < len(full_sequence) and i < len(sequence):  # 确保索引在范围内
                    cache_position = sequence[i] + 1  # 0-3 变成 1-4
                    full_sequence[idx] = cache_position

            # 实际使用的位置就是sequence中的值+1
            used_positions = [val + 1 for val in sequence]  # 0-3 变成 1-4

            logger.info(f"interval_seq: {self.interval_seq}")
            logger.info(f"生成的缓存随机序列: {sequence}")
            logger.info(f"生成的100时间步序列: {full_sequence}")
            logger.info(f"实际使用的缓存位置: {used_positions}")
            """            
            对比维度	full_sequence	                sequence	                   used_positions
            长度	        100	                                4 (num_positions)	       4(num_positions)
            值范围	     0-4	                                0-3	                            1-4
            含义	完整的100步调度	                    在间隔位置的原始选择	    实际使用的缓存模型编号
            用途	供进化算法使用	                        中间计算变量	                记录实际缓存模型
            示例	[0,0,0,...,3,...,1,...,4,...,2]	                [2, 0, 3, 1]	                    [3, 1, 4, 2]
            """
            # 返回三个值：完整序列、实际使用的位置和原始序列
            return full_sequence, used_positions, sequence

    def _calculate_external_fid(self):
        """调用外部FID计算命令"""
        import subprocess
        import re

        # FID计算命令
        fid_command = [
            "python", "./evaluations/fid_metric/fid_score.py",
            "./dataset/cifar10-sample-image/",
            "./evaluations/fid_statistic/fid_stats_cifar10.npz"
        ]

        logger.info(f"🔍 执行FID计算命令: {' '.join(fid_command)}")

        try:
            # 执行FID计算命令
            result = subprocess.run(fid_command, capture_output=True, text=True, check=True)

            # 解析FID分数
            fid_output = result.stdout.strip()
            logger.info(f"FID计算输出: {fid_output}")

            # 从输出中提取FID分数
            # 通常格式是: "FID: 45.67" 或直接输出数字
            fid_match = re.search(r'FID:\s*([\d.]+)', fid_output)
            if fid_match:
                fid_score = float(fid_match.group(1))
            else:
                # 如果没有匹配到，尝试直接转换最后一行
                lines = fid_output.split('\n')
                last_line = lines[-1].strip()
                try:
                    fid_score = float(last_line)
                except ValueError:
                    logger.error(f"无法解析FID分数，输出: {fid_output}")
                    fid_score = 100.0  # 默认值

            logger.info(f"📊 解析到的FID分数: {fid_score:.4f}")
            return fid_score

        except subprocess.CalledProcessError as e:
            logger.error(f"FID计算命令执行失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            return 100.0  # 错误时的默认值
        except Exception as e:
            logger.error(f"FID计算过程中发生错误: {e}")
            return 100.0  # 错误时的默认值
    # def _save_to_dataset(self, cache_sequence, fid_score):
    #     """保存数据到数据集"""
    #     try:
    #         dataset_dir = "dataset/cifar10-dataset"
    #         os.makedirs(dataset_dir, exist_ok=True)
    #
    #         # 找到下一个可用的文件编号
    #         existing_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pth')]
    #         existing_indices = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
    #         next_index = max(existing_indices) + 1 if existing_indices else 0
    #
    #         filename = f"{next_index}.pth"
    #         filepath = os.path.join(dataset_dir, filename)
    #
    #         # 按照OMS-DPM标准格式保存
    #         data = {
    #             'ms': cache_sequence,  # 缓存序列作为模型调度
    #             'metric': fid_score  # FID分数
    #         }
    #
    #         torch.save(data, filepath)
    #         logger.info(f"💾 数据保存到: {filepath}")
    #
    #         # 记录详细信息 .pth
    #         self._log_dataset_entry(cache_sequence, fid_score, filename)
    #
    #     except Exception as e:
    #         logger.error(f"❌ 保存数据失败: {e}")

    def _save_to_dataset(self, cache_sequence, fid_score, file_index):
        """保存数据到数据集 - 将所有数据保存到单个文件"""
        try:
            dataset_dir = "dataset/cifar10-dataset"
            os.makedirs(dataset_dir, exist_ok=True)

            # 主数据集文件名
            main_filename = "cifar10_ddim_dataset.pth"
            main_filepath = os.path.join(dataset_dir, main_filename)

            # 按照OMS-DPM标准格式创建数据项
            data_item = {
                'model_schedule': cache_sequence,  # 缓存序列作为模型调度
                'score': fid_score  # FID分数
            }

            # 如果文件已存在，则加载并追加数据
            if os.path.exists(main_filepath):
                existing_data = torch.load(main_filepath, map_location='cpu')
                if not isinstance(existing_data, list):
                    # 如果现有数据不是列表，转换为列表
                    existing_data = [existing_data]
                existing_data.append(data_item)
                torch.save(existing_data, main_filepath)
            else:
                # 创建新文件
                torch.save([data_item], main_filepath)

            logger.info(
                f"💾 数据已追加到: {main_filepath} (总条目数: {len(torch.load(main_filepath)) if os.path.exists(main_filepath) else 1})")

            # 记录详细信息
            self._log_dataset_entry(cache_sequence, fid_score, file_index)

            # 可选：保存单个文件备份
            backup_dir = os.path.join(dataset_dir, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            backup_filepath = os.path.join(backup_dir, f"{file_index}.pth")
            torch.save(data_item, backup_filepath)

        except Exception as e:
            logger.error(f"❌ 保存数据失败: {e}")

    def _save_to_dataset_population(self, cache_sequence, fid_score,population_file_result_folder=None,population_file_name = None):
        """保存数据到数据集 - 将所有数据保存到单个文件"""
        # 按照OMS-DPM标准格式创建数据项
        data_item = {
            'model_schedule': cache_sequence,  # 缓存序列作为模型调度
            'score': fid_score  # FID分数
        }
        # 可选：保存单个文件备份
        backup_filepath = os.path.join(population_file_result_folder,f"{population_file_name}.pth")
        torch.save(data_item, backup_filepath)

    def _save_cache_positions_to_log(self, used_positions):
        """保存缓存位置到日志文件 - 使用实际使用的位置"""
        log_file = "random_cache_position_fid_score.log"

        position_descriptions = {
            1: "up3.block1",
            2: "up2.block1",
            3: "up1.block1",
            4: "up0.block1"
        }

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        log_content = f"[{current_time}] 实际使用的缓存位置:\n"
        for i, pos in enumerate(used_positions):
            desc = position_descriptions[pos]
            log_content += f"  缓存点{i:2d}: {pos}-{desc}\n"
        log_content += "\n"

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_content)
            logger.info(f"缓存位置已保存到: {log_file}")
        except Exception as e:
            logger.error(f"保存缓存位置日志失败: {e}")

    def _save_time_taken_to_log(self, total_time):
        """记录时间到日志文件"""
        try:
            # 使用指定的日志文件名在当前目录
            log_filename = "random_cache_position_fid_score.log"

            # 记录时间信息
            time_entry = f"Time taken: {total_time} seconds\n"

            # 追加写入日志文件
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(time_entry)

            logger.info(f"⏱️ 时间已记录到日志: {log_filename}")

        except Exception as e:
            logger.error(f"❌ 记录时间日志失败: {e}")

    # def _log_dataset_entry(self, full_sequence, fid_score, filename):
    #     """记录数据集条目.pth - 使用完整的100时间步序列"""
    #     log_file = "dataset_generation.log"
    #
    #     position_descriptions = {
    #         0: "使用缓存",
    #         1: "up3.block1",
    #         2: "up2.block1",
    #         3: "up1.block1",
    #         4: "up0.block1"
    #     }
    #
    #     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #
    #     # 计算统计信息
    #     position_counts = {}
    #     for pos in full_sequence:
    #         position_counts[pos] = position_counts.get(pos, 0) + 1
    #
    #     log_content = f"[{current_time}] {filename}\n"
    #     log_content += f"  FID: {fid_score:.4f}\n"
    #     log_content += f"  完整100时间步序列: {full_sequence}\n"
    #     log_content += "  位置统计:\n"
    #
    #     for pos in sorted(position_counts.keys()):
    #         count = position_counts[pos]
    #         percentage = (count / len(full_sequence)) * 100
    #         desc = position_descriptions[pos]
    #         log_content += f"    {pos}-{desc}: {count}次 ({percentage:.1f}%)\n"
    #
    #     log_content += "\n"
    #
    #     try:
    #         with open(log_file, "a", encoding="utf-8") as f:
    #             f.write(log_content)
    #     except Exception as e:
    #         logger.error(f"保存日志失败: {e}")
    def _log_dataset_entry(self, full_sequence, fid_score, file_index):
        """记录数据集条目 - 使用完整的100时间步序列"""
        log_file = "dataset_generation.log"

        position_descriptions = {
            0: "使用缓存",
            1: "up3.block1",
            2: "up2.block1",
            3: "up1.block1",
            4: "up0.block1"
        }

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 计算统计信息
        position_counts = {}
        for pos in full_sequence:
            position_counts[pos] = position_counts.get(pos, 0) + 1

        log_content = f"[{current_time}] 条目索引: {file_index}\n"
        log_content += f"  FID: {fid_score:.4f}\n"
        log_content += f"  完整100时间步序列: {full_sequence}\n"
        log_content += "  位置统计:\n"

        for pos in sorted(position_counts.keys()):
            count = position_counts[pos]
            percentage = (count / len(full_sequence)) * 100
            desc = position_descriptions[pos]
            log_content += f"    {pos}-{desc}: {count}次 ({percentage:.1f}%)\n"

        log_content += f"  已保存到: dataset/cifar10-dataset/cifar10_ddim_dataset.pth (作为第{file_index}个条目)\n"
        log_content += "\n"

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_content)
        except Exception as e:
            logger.error(f"保存日志失败: {e}")

    # def generate_dataset(self):
    #     """生成预测器训练数据集"""
    #     import shutil
    #     import os
    #     import time
    #     from tqdm import tqdm
    #     import logging
    #
    #     # 临时关闭控制台日志
    #     original_level = logging.getLogger().getEffectiveLevel()
    #     logging.getLogger().setLevel(logging.WARNING)
    #
    #     try:
    #         # 导入缓存调度
    #         current_dir = os.path.dirname(os.path.abspath(__file__))
    #         parent_dir = os.path.dirname(current_dir)
    #         utils_dir = os.path.join(parent_dir, "utils")
    #         import sys
    #         if utils_dir not in sys.path:
    #             sys.path.insert(0, utils_dir)
    #         from cache_schedule_util import get_cache_schedule
    #
    #         # 创建目录
    #         os.makedirs(self.config.dataset_generate.dataset_path, exist_ok=True)
    #
    #         # 加载模型
    #         model = self.creat_model(dataset_generate_mode=True)
    #
    #         total_files = self.config.dataset_generate.data_num
    #         print(f"开始生成 {total_files} 个文件...")
    #
    #         total_start = time.time()
    #         fids = []
    #
    #         with tqdm(range(total_files), desc="进度", unit="文件") as pbar:
    #             for i in pbar:
    #                 start_time = time.time()
    #
    #                 # 生成缓存序列
    #                 import random
    #                 import numpy as np
    #                 original_state = random.getstate()
    #                 original_np_state = np.random.get_state()
    #
    #                 random.seed(self.args.seed + i * 1000)
    #                 np.random.seed(self.args.seed + i * 1000)
    #
    #                 cache_seq = get_cache_schedule(
    #                     self.config.dataset_generate.cache_schedule,
    #                     self.args.timesteps,
    #                     num_cache_positions=4
    #                 )
    #
    #                 random.setstate(original_state)
    #                 np.random.set_state(original_np_state)
    #
    #                 # 确保目录存在
    #                 os.makedirs(self.args.image_folder, exist_ok=True)
    #
    #                 # 生成图像
    #                 self.args.dataset_generate = True
    #                 self.dataset_generate_sampling(
    #                     model, cache_seq,
    #                     total_n_samples=self.config.dataset_generate.image_num,
    #                     save_images=True, classifier=None
    #                 )
    #
    #                 # 计算FID
    #                 fid = self._calculate_external_fid()
    #                 fids.append(fid)
    #
    #                 # 保存数据
    #                 self._save_to_dataset(cache_seq, fid)
    #                 self._log_dataset_entry(cache_seq, fid, f"{i}.pth")
    #
    #                 # 清理
    #                 if os.path.exists(self.args.image_folder):
    #                     shutil.rmtree(self.args.image_folder)
    #
    #                 # 更新进度条
    #                 file_time = time.time() - start_time
    #                 pbar.set_postfix({
    #                     "FID": f"{fid:.1f}",
    #                     "时间": f"{file_time:.0f}s"
    #                 })
    #
    #         # 恢复日志
    #         logging.getLogger().setLevel(original_level)
    #
    #         # 显示结果
    #         total_time = time.time() - total_start
    #         print(f"\n✅ 完成! 共 {total_files} 个文件")
    #         print(f"📊 平均FID: {sum(fids) / len(fids):.2f}")
    #         print(f"⏱️  总耗时: {total_time / 60:.1f}分钟")
    #
    #         return True
    #
    #     except Exception as e:
    #         logging.getLogger().setLevel(original_level)
    #         print(f"❌ 错误: {e}")
    #         return False
    def generate_dataset(self):
        """生成预测器训练数据集"""
        import shutil
        import os
        import time
        from tqdm import tqdm
        import logging

        # 临时关闭控制台日志
        original_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.WARNING)

        try:
            # 导入缓存调度
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            utils_dir = os.path.join(parent_dir, "utils")
            import sys
            if utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            from cache_schedule_util import get_cache_schedule

            # 创建目录
            dataset_dir = self.config.dataset_generate.dataset_path
            os.makedirs(dataset_dir, exist_ok=True)

            # 主数据集文件路径
            main_filepath = os.path.join(dataset_dir, "cifar10_ddim_dataset.pth")

            # 检查是否已存在数据集文件
            if os.path.exists(main_filepath):
                existing_data = torch.load(main_filepath, map_location='cpu')
                start_index = len(existing_data) if isinstance(existing_data, list) else 1
                print(f"📂 检测到现有数据集文件，包含 {start_index} 个条目")
            else:
                start_index = 0
                print("📂 创建新的数据集文件")

            # 加载模型
            model = self.creat_model(dataset_generate_mode=True)

            total_files = self.config.dataset_generate.data_num
            print(f"开始生成 {total_files} 个文件...")

            total_start = time.time()
            fids = []

            with tqdm(range(total_files), desc="进度", unit="文件") as pbar:
                for i in pbar:
                    file_index = start_index + i
                    start_time = time.time()

                    cache_seq = get_cache_schedule(
                        self.config.dataset_generate.cache_schedule,
                        self.args.timesteps,
                        num_cache_positions=4
                    )

                    # 确保目录存在
                    os.makedirs(self.args.image_folder, exist_ok=True)

                    # 生成图像
                    self.args.dataset_generate = True
                    self.dataset_generate_sampling(
                        model, cache_seq,
                        total_n_samples=self.config.dataset_generate.image_num,
                        save_images=True, classifier=None
                    )

                    # 计算FID
                    fid = self._calculate_external_fid()
                    fids.append(fid)

                    # 保存数据到单个文件
                    self._save_to_dataset(cache_seq, fid, file_index)

                    # 清理
                    if os.path.exists(self.args.image_folder):
                        shutil.rmtree(self.args.image_folder)

                    # 更新进度条
                    file_time = time.time() - start_time
                    pbar.set_postfix({
                        "FID": f"{fid:.1f}",
                        "索引": f"{file_index}",
                        "时间": f"{file_time:.0f}s"
                    })

                    # 定期保存检查点（每100个条目）
                    if (i + 1) % 100 == 0:
                        # 重新加载并显示当前数据集状态
                        if os.path.exists(main_filepath):
                            data = torch.load(main_filepath, map_location='cpu')
                            print(f"✅ 检查点保存: 数据集目前包含 {len(data)} 个条目")

            # 恢复日志
            logging.getLogger().setLevel(original_level)

            # 显示最终结果
            total_time = time.time() - total_start
            print(f"\n✅ 完成! 共生成 {total_files} 个条目")
            print(f"📊 平均FID: {sum(fids) / len(fids):.2f}")
            print(f"⏱️  总耗时: {total_time / 60:.1f}分钟")

            # 最终数据集信息
            if os.path.exists(main_filepath):
                data = torch.load(main_filepath, map_location='cpu')
                print(f"📁 数据集文件: {main_filepath}")
                print(f"📊 总条目数: {len(data)}")
                print(f"💾 文件大小: {os.path.getsize(main_filepath) / (1024 * 1024):.2f} MB")

            return True

        except Exception as e:
            logging.getLogger().setLevel(original_level)
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    def dataset_generate_sampling(self, model, cache_sequence, total_n_samples=50000, save_images=True, classifier=None,
                                  timesteps=None):
        """
        数据集生成专用的采样函数
        对应 sample_image_position_matrix，但简化了逻辑
        """
        config = self.config
        img_id = 0
        logger.info(f"starting from image {img_id}")

        total_n_samples = total_n_samples // self.accelerator.num_processes
        n_rounds = (total_n_samples - img_id) // self.args.sample_batch

        generate_samples = []
        throughput = []
        sample_start_time = time.time()

        with torch.no_grad(), tqdm.tqdm(range(n_rounds)) as t:
            for _ in t:
                start_time = time.time()
                n = self.args.sample_batch

                # 生成噪声（与sample_image_position_matrix相同）
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                # 采样逻辑（与sample_image_position_matrix类似）
                if timesteps is None:
                    timesteps = self.args.timesteps

                if self.args.sample_type == "generalized":
                    if self.args.skip_type == "uniform":
                        skip = self.num_timesteps // timesteps
                        seq = range(0, self.num_timesteps, skip)
                    elif self.args.skip_type == "quad":
                        seq = (
                                np.linspace(
                                    0, np.sqrt(self.num_timesteps * 0.8), timesteps
                                )
                                ** 2
                        )
                        seq = [int(s) for s in list(seq)]
                    else:
                        raise NotImplementedError

                    # 使用数据集生成专用的采样函数
                    from ..functions.deepcache_denoising import dataset_generate_steps

                    xs = dataset_generate_steps(
                        x, seq, model, self.betas,
                        timesteps=timesteps,
                        cache_sequence=cache_sequence,
                        eta=self.args.eta,
                        args=self.args
                    )

                    x = xs[0][-1]  # 取最后一个时间步

                elif self.args.sample_type == "ddpm_noisy":
                    # 数据集生成模式不支持的采样类型
                    raise NotImplementedError("数据集生成模式不支持 ddpm_noisy 采样")
                else:
                    raise NotImplementedError(f"不支持的采样类型: {self.args.sample_type}")

                x = inverse_data_transform(config, x)

                use_time = time.time() - start_time
                throughput.append(x.shape[0] / use_time)
                t.set_description(f"Throughput: {np.mean(throughput):.2f} samples/s")

                if save_images:
                    for i in range(n):
                        tvu.save_image(
                            x[i], os.path.join(self.args.image_folder, f"{self.accelerator.process_index}_{img_id}.png")
                        )
                        img_id += 1
                else:
                    generate_samples.append(x)

        self.args.accelerator.wait_for_everyone()
        total_time = time.time() - sample_start_time
        logger.info(f"图像生成耗时: {total_time:.2f} 秒")

        return generate_samples


        
