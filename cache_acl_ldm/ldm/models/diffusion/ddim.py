"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F
import time

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import logging
logger = logging.getLogger(__name__)  # 获取 ddim.py 的 logger
from quant.utils import seed_everything


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", slow_steps=None, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.slow_steps = slow_steps
        self.quant_sample = False
        self.args = None

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)
        # make_ddim_timesteps 来自modules/diffusionmodules/util.py
    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose) # self.ddim_timesteps = [1, 5, 9, ..., 997]（250个步）
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               replicate_interval=None,
               nonuniform=False, pow=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        # 计算DDIM需要的所有参数：时间步、alpha、sigma等
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        #print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    replicate_interval=replicate_interval,
                                                    nonuniform=nonuniform, pow=pow,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      replicate_interval=None, nonuniform=False, pow=None):
        device = self.model.betas.device
        b = shape[0] # batch_size = 25
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps # 这里决定timesteps是数字还是列表 | self.ddim_timesteps = [1, 5, 9, ..., 997]（250个步）
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        # 保存中间结果用于调试
        intermediates = {'x_inter': [img.to('cpu')], 'pred_x0': [img.to('cpu')], 'ts': [], 'cond': [], 'uncond': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0] # total_steps = 250
        #print(f"Running DDIM Sampling with {total_steps} timesteps")
        # total=self.ddpm_num_timesteps
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True) #

        self.model.model.reset_current_t()
        if self.slow_steps is None:
            self.model.model.set_interval(total_steps, replicate_interval, nonuniform=nonuniform, pow=pow)
        else:
            self.model.model.slow_steps = self.slow_steps
        print("缓存时间步为：", self.model.model.slow_steps)
        """从这里开始完整的进行循环的单步去噪"""
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            if self.quant_sample:# set time steps quantizer
                self.model.model.diffusion_model.set_time(i)
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)

            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            # # if index % log_every_t == 0 or index == total_steps - 1:
            # intermediates['x_inter'].append(img.to('cpu'))
            # # intermediates['pred_x0'].append(pred_x0.to('cpu'))
            # intermediates['ts'].append(ts.to('cpu'))
            # intermediates['cond'].append(cond.to('cpu'))
            # intermediates['uncond'].append(unconditional_conditioning.to('cpu'))

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device
        # p_sample_ddim()-单步去噪
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: ## do not run
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample_position_matrix(self,
                               S,
                               batch_size,
                               shape,
                               conditioning=None,
                               callback=None,
                               normals_sequence=None,
                               img_callback=None,
                               quantize_x0=False,
                               eta=0.,
                               mask=None,
                               x0=None,
                               temperature=1.,
                               noise_dropout=0.,
                               score_corrector=None,
                               corrector_kwargs=None,
                               verbose=True,
                               x_T=None,
                               log_every_t=100,
                               unconditional_guidance_scale=1.,
                               unconditional_conditioning=None,
                               replicate_interval=None,
                               nonuniform=False, pow=None,
                               cache_sequence=None,
                               full_cache_sequence=None,
                               args = None
                               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
                               # **kwargs
                               ):
        """我们这里使用不同的分辨率，对应了不同层的缓存位置-4resolution，使用我们自己预定义的缓存位置"""
        # === 预生成缓存位置序列 ===
        self.args = args
        # seed_everything(self.args.seed)
        if self.args.random_cache:
            # 分支1: 使用固定缓存位置 (--cache_position)
            if hasattr(self.args, 'cache_position') and self.args.cache_position is not None:
                # 固定位置模式：所有缓存点使用相同的位置
                cache_position = self.args.cache_position
                logger.info(f"使用固定缓存位置模式: position={cache_position}")

                self.full_cache_sequence, self.used_cache_sequence, self.cache_sequence = self._generate_cache_sequence(
                     len(self.slow_steps), fixed_position=cache_position
                )

            # 分支2: 使用预设的非单一缓存位置 (--none_cache_position)
            else:
                # 预设序列模式：直接使用外部传入的缓存序列
                # logger.info("使用预设的非单一缓存位置序列")
                # 确保传入的参数正确
                if cache_sequence is None:
                    raise ValueError("预设缓存序列模式需要传入 cache_sequence 参数")
                if full_cache_sequence is None:
                    raise ValueError("预设缓存序列模式需要传入 full_cache_sequence 参数")

                self.cache_sequence = cache_sequence
                # self.cache_sequence = [3,0,0,0,0,0,0,3,3,3] 这里缓存代码里面使用的缓存索引，默认0开始
                self.full_cache_sequence = full_cache_sequence
                self.used_cache_sequence = [x + 1 for x in self.cache_sequence]
                # self.used_cache_sequence

                # 验证序列长度
                if len(self.cache_sequence) != len(self.slow_steps):
                    logger.warning(
                        f"缓存序列长度不匹配: sequence_len={len(self.cache_sequence)}, interval_len={len(self.slow_steps)}")

            # 分支3: 默认生成随机缓存位置
            # else:
            #     # 随机位置模式：每个缓存点随机选择位置
            #     logger.info("使用随机缓存位置模式")
            #
            #     self.full_cache_sequence, self.used_cache_sequence, self.cache_sequence = self._generate_cache_sequence(
            #          len(self.slow_steps)
            #     )
        """" --------------------分割线，上面是新增代码-------------------"""
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        # 计算DDIM需要的所有参数：时间步、alpha、sigma等  # self.ddim_timesteps = [1, 5, 9, ..., 997]（250个步）
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling_postition_matrix(conditioning, size,
                                                                     callback=callback,
                                                                     img_callback=img_callback,
                                                                     quantize_denoised=quantize_x0,
                                                                     mask=mask, x0=x0,
                                                                     ddim_use_original_steps=False,
                                                                     noise_dropout=noise_dropout,
                                                                     temperature=temperature,
                                                                     score_corrector=score_corrector,
                                                                     corrector_kwargs=corrector_kwargs,
                                                                     x_T=x_T,
                                                                     log_every_t=log_every_t,
                                                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                                                     unconditional_conditioning=unconditional_conditioning,
                                                                     replicate_interval=replicate_interval,
                                                                     nonuniform=nonuniform, pow=pow,
                                                                     cache_sequence=self.cache_sequence
                                                                     )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling_postition_matrix(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      replicate_interval=None, nonuniform=False, pow=None,cache_sequence=None):
        device = self.model.betas.device
        b = shape[0] # batch_size = 25
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps # 这里决定timesteps是数字还是列表 | self.ddim_timesteps = [1, 5, 9, ..., 997]（250个步）
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        # 保存中间结果用于调试
        intermediates = {'x_inter': [img.to('cpu')], 'pred_x0': [img.to('cpu')], 'ts': [], 'cond': [], 'uncond': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0] # total_steps = 250
        #print(f"Running DDIM Sampling with {total_steps} timesteps")
        # total=self.ddpm_num_timesteps
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True) #

        self.model.model.reset_current_t() #重置250步时间步
        self.model.model.reset_current_cache_index() #重置缓存索引
        if self.slow_steps is None:
            self.model.model.set_interval(total_steps, replicate_interval, nonuniform=nonuniform, pow=pow)
        else:
            self.model.model.slow_steps = self.slow_steps    #后续收集数据的时候，更改这里的代码即可
        print("缓存时间步为：", self.model.model.slow_steps)
        """从这里开始完整的进行循环的单步去噪"""
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            if self.quant_sample:# set time steps quantizer
                self.model.model.diffusion_model.set_time(i)
            # 开始一个批次的去噪，返回一个批次的干净图片
            outs = self.p_sample_ddim_position_matrix(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      cache_sequence=cache_sequence                )

            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            # # if index % log_every_t == 0 or index == total_steps - 1:
            # intermediates['x_inter'].append(img.to('cpu'))
            # # intermediates['pred_x0'].append(pred_x0.to('cpu'))
            # intermediates['ts'].append(ts.to('cpu'))
            # intermediates['cond'].append(cond.to('cpu'))
            # intermediates['uncond'].append(unconditional_conditioning.to('cpu'))
            """这里img也就是一个批次去噪之后的图片，例如自定义的batch-25张"""
        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim_position_matrix(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,cache_sequence=None):
        b, *_, device = *x.shape, x.device
        # p_sample_ddim()-单步去噪
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model_position_matrix(x, t, c,cache_sequence=cache_sequence) #传入我们自定义的缓存更新序列
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model_position_matrix(x_in, t_in, c_in,cache_sequence=cache_sequence).chunk(2) #传入我们自定义的缓存更新序列
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: ## do not run
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    """ only simple random / cache_schedule_util.py """
    def _generate_cache_sequence(self, num_positions, fixed_position=None):
        """根据interval_seq生成250个时间步的序列"""
        import random

        # 定义总时间步数
        total_timesteps = 250  # 修改为250步

        if fixed_position is not None:
            # 使用固定位置模式
            logger.info(f"使用固定缓存位置: {fixed_position}")
            logger.info(f"总时间步数: {total_timesteps}, interval_seq: {self.slow_steps}")

            # 初始化250个时间步，全部设为0（使用缓存）
            full_sequence = [0] * total_timesteps

            # 生成固定位置的序列（所有位置都使用相同的固定值）
            sequence = [fixed_position - 1 for _ in range(num_positions)]  # 1-4 变成 0-3

            # 在interval_seq指定的索引位置设置缓存位置
            for i, idx in enumerate(self.slow_steps):
                if idx < len(full_sequence) and i < len(sequence):  # 确保索引在范围内
                    cache_position = sequence[i] + 1  # 0-3 变成 1-4
                    full_sequence[idx] = cache_position

            # 实际使用的位置就是固定位置
            used_positions = [fixed_position] * num_positions

            logger.info(f"生成的固定缓存序列: {sequence[:10]}...")  # 只显示前10个
            logger.info(f"生成的{total_timesteps}时间步序列 (示例): {full_sequence[:20]}...")  # 只显示前20个
            logger.info(f"实际使用的缓存位置: {used_positions[:10]}...")  # 只显示前10个

            # 返回三个值：完整序列、实际使用的位置和原始序列
            return full_sequence, used_positions, sequence

        else:
            # 原有的随机生成逻辑
            logger.info(f"总时间步数: {total_timesteps}, interval_seq: {self.slow_steps}")

            # 初始化250个时间步，全部设为0（使用缓存）
            full_sequence = [0] * total_timesteps

            # 生成索引序列 [0, 1, 2, 3] 的随机组合（四个位置）
            sequence = [random.randint(0, 3) for _ in range(num_positions)]

            # 在interval_seq指定的索引位置设置缓存位置，使用sequence中的值+1（变成1-4）
            for i, idx in enumerate(self.slow_steps):
                if idx < len(full_sequence) and i < len(sequence):  # 确保索引在范围内
                    cache_position = sequence[i] + 1  # 0-3 变成 1-4
                    full_sequence[idx] = cache_position

            # 实际使用的位置就是sequence中的值+1
            used_positions = [val + 1 for val in sequence]  # 0-3 变成 1-4

            logger.info(f"生成的缓存随机序列: {sequence}")
            logger.info(f"生成的{total_timesteps}时间步序列 (示例): {full_sequence}")
            logger.info(f"实际使用的缓存位置: {used_positions}")  # 只显示前10个
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

    def _save_cache_positions_to_log(self, used_positions):
        """保存缓存位置到日志文件 - 使用实际使用的位置"""
        log_file = "imagenet_random_cache_position_fid_score.log"

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


class DDIMSampler_trainer(object):
    def __init__(self, model, quant_model, lr_scheduler, optimizer, schedule="linear", slow_steps=None, **kwargs):
        super().__init__()
        self.model = model
        self.quant_model = quant_model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.slow_steps = slow_steps
        self.quant_sample = False
        self.loss = []

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               replicate_interval=None,
               nonuniform=False, pow=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        #print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    replicate_interval=replicate_interval,
                                                    nonuniform=nonuniform, pow=pow,
                                                    )
        return samples, intermediates

    # @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      replicate_interval=None, nonuniform=False, pow=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img.to('cpu')], 'pred_x0': [img.to('cpu')], 'ts': [], 'cond': [], 'uncond': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        #print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True)

        self.model.model.reset_current_t()
        self.model.model.slow_steps = list(range(0, total_steps))
        self.quant_model.model.reset_current_t()
        if self.slow_steps is None:
            self.quant_model.model.set_interval(total_steps, replicate_interval, nonuniform=nonuniform, pow=pow)
        else:
            self.quant_model.model.slow_steps = self.slow_steps
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            if self.quant_sample:# set time steps quantizer
                self.quant_model.model.diffusion_model.set_time(i)
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            # # if index % log_every_t == 0 or index == total_steps - 1:
            # intermediates['x_inter'].append(img.to('cpu'))
            # # intermediates['pred_x0'].append(pred_x0.to('cpu'))
            # intermediates['ts'].append(ts.to('cpu'))
            # intermediates['cond'].append(cond.to('cpu'))
            # intermediates['uncond'].append(unconditional_conditioning.to('cpu'))

        return img, intermediates

    # @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device
        self.optimizer.zero_grad()

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            quant_e_t = self.quant_model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2).detach()
            t_in = torch.cat([t] * 2).detach()
            c_in = torch.cat([unconditional_conditioning, c]).detach()

            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            quant_e_t_uncond, quant_e_t = self.quant_model.apply_model(x_in, t_in, c_in).chunk(2)
            quant_e_t = quant_e_t_uncond + unconditional_guidance_scale * (quant_e_t - quant_e_t_uncond)

        loss = F.mse_loss(quant_e_t, e_t, size_average=False)
        loss.backward()
        self.loss.append(loss.detach())
        self.optimizer.step()
        self.lr_scheduler.step()

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: ## do not run
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
