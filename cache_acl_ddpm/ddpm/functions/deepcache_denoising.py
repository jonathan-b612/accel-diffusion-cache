import torch

from scipy.stats import shapiro
import numpy as np

def sample_gaussian_centered(n=1000, sample_size=100, std_dev=100, shift=0):
    samples = []
    
    while len(samples) < sample_size:
        # Sample from a Gaussian centered at n/2
        sample = int(np.random.normal(loc=n/2+shift, scale=std_dev))
        
        # Check if the sample is in bounds
        if 1 <= sample < n and sample not in samples:
            samples.append(sample)
    
    return samples

def sample_from_quad_center(total_numbers, n_samples, center, pow=1.2):
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace((-center)**(1/pow), (total_numbers-center)**(1/pow), n_samples+1)
        #print(x_values)
        #print([x for x in np.unique(np.int32(x_values**pow))[:-1]])
        # Raise these values to the power of 1.5 to get a non-linear distribution
        indices = [0] + [x+center for x in np.unique(np.int32(x_values**pow))[1:-1]]
        if len(indices) == n_samples:
            break
        
        pow -=0.02
    return indices, pow

def sample_from_quad(total_numbers, n_samples, pow=1.2):
    # Generate linearly spaced values between 0 and a max value
    x_values = np.linspace(0, total_numbers**(1/pow), n_samples+1)

    # Raise these values to the power of 1.5 to get a non-linear distribution
    indices = np.unique(np.int32(x_values**pow))[:-1]
    return indices

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, timesteps, cache_interval=None, non_uniform=False, pow=None, center=None,  branch=None, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        prv_f = None

        cur_i = 0
        if non_uniform:
            num_slow = timesteps // cache_interval
            if timesteps % cache_interval > 0:
                num_slow += 1
            interval_seq, final_pow = sample_from_quad_center(total_numbers=timesteps, n_samples=num_slow, center=center, pow=pow)
        else:
            interval_seq = list(range(0, timesteps, cache_interval))
            interval = cache_interval
        #print(non_uniform, interval_seq)
        

        slow_path_count = 0
        save_features = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            with torch.no_grad():
                if cur_i in interval_seq: #%
                #if cur_i % interval == 0:
                    #print(cur_i, interval_seq)
                    et, cur_f = model(xt, t, prv_f=None, context=None,branch=branch)
                    prv_f = cur_f[0]
                    save_features.append(cur_f[0].detach().cpu())
                    slow_path_count+= 1
                else:
                    et, cur_f = model(xt, t, prv_f=prv_f, context=None,branch=branch)
                    #quick_path_count+= 1

            #print(i, torch.mean(et) / torch.mean(xt), torch.var(et)/torch.var(xt), torch.mean(et), torch.var(et))

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

            cur_i += 1

    return xs, x0_preds

def adaptive_generalized_steps(x, seq, model, b, timesteps, interval_seq=None, branch=None, quant=False, **kwargs):
    print(f"=== 进入 adaptive_generalized_steps ===")
    print(f"interval_seq: {interval_seq}")
    print(f"kwargs keys: {kwargs.keys()}")
    if 'args' in kwargs and kwargs['args'] is not None:
        print(f"random_cache: {getattr(kwargs['args'], 'random_cache', '未设置')}")
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        prv_f = None
        cur_i = 0
        # 获取缓存序列
        cache_sequence = kwargs.get('cache_sequence')
        use_fixed_cache = (cache_sequence is not None and
                           kwargs.get('args') is not None and
                           kwargs['args'].random_cache)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            print(f"时间步: {i} -> {j}, prv_f: {'有' if prv_f is not None else '无'}")
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda') # xs[-1] 就是当前循环中要处理的图像,step by step
            if quant:
                time = len(xs) - 1
                model.set_time(time)
            if cur_i in interval_seq:
                print(f"时间步 {cur_i} 在 interval_seq 中，生成新缓存")
                # === 关键修改：在interval_seq时间步设置随机缓存 ===
                if hasattr(model, 'set_random_cache_para') and kwargs.get('args') is not None and kwargs['args'].random_cache:
                # === 增加：如果提供了缓存序列，则设置 ===
                    cache_sequence = kwargs.get('cache_sequence')
                    if cache_sequence is not None:
                        cache_idx = interval_seq.index(cur_i)  # 当前是第几个缓存点
                        if cache_idx < len(cache_sequence):
                            cache_index = cache_sequence[cache_idx]
                            print(f"！！！！！！！！！！！！！！使用预定义缓存索引: {cache_index}")
                            # 重新设置缓存位置，使用预定义索引
                            model.set_random_cache_para(cache_index=cache_index)
                    else:
                        print("不使用预定义缓存序列")
                        model.set_random_cache_para()  # 随机选择缓存位置
                et, cur_f = model(xt, t, context=None, prv_f=None, branch=branch)
                print(f"cur_f长度: {len(cur_f) if hasattr(cur_f, '__len__') else 'N/A'}")
                if cur_f is not None and hasattr(cur_f, '__len__'):
                    for i, f in enumerate(cur_f):
                        print(f"cur_f[{i}]: {f.shape if f is not None else None}")

                # 检查cur_f[0]是否有效
                if cur_f and len(cur_f) > 0 and cur_f[0] is not None:
                    prv_f = cur_f[0]
                    print(f"✅ 设置prv_f: {prv_f.shape}")
            else:
                print(f"时间步 {cur_i} 不在 interval_seq 中，使用缓存")
                et, cur_f = model(xt, t, context=None, prv_f=prv_f, branch=branch)

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
            cur_i += 1

    return xs, x0_preds


# def adaptive_generalized_steps(x, seq, model, b, timesteps, interval_seq=None, branch=None, quant=False, **kwargs):
#     print(f"=== 进入 adaptive_generalized_steps ===")
#     print(f"interval_seq: {interval_seq}")
#
#     # 添加时间记录
#     import time
#     timing_data = {
#         'generate_cache_times': [],
#         'use_cache_times': [],
#         'current_cache_depth': 0,
#         'step_cache_depths': [],
#         'cache_sequence': kwargs.get('cache_sequence', []),
#         'interval_seq': interval_seq,
#         'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
#     }
#
#     if 'args' in kwargs and kwargs['args'] is not None:
#         print(f"random_cache: {getattr(kwargs['args'], 'random_cache', '未设置')}")
#
#     with torch.no_grad():
#         n = x.size(0)
#         seq_next = [-1] + list(seq[:-1])
#         x0_preds = []
#         xs = [x]
#         prv_f = None
#         cur_i = 0
#
#         cache_sequence = kwargs.get('cache_sequence')
#         use_fixed_cache = (cache_sequence is not None and
#                            kwargs.get('args') is not None and
#                            kwargs['args'].random_cache)
#
#         # === 预热：先运行几次让GPU稳定 ===
#         print("正在预热GPU...")
#         for warmup in range(5):
#             dummy_input = torch.randn_like(x)
#             dummy_t = torch.ones(n).to(x.device) * 1000
#             _ = model(dummy_input, dummy_t, context=None, prv_f=None, branch=branch)
#         torch.cuda.synchronize()  # 确保预热完成
#         print("GPU预热完成")
#
#         for i, j in zip(reversed(seq), reversed(seq_next)):
#             print(f"时间步: {i} -> {j}, prv_f: {'有' if prv_f is not None else '无'}")
#
#             t = (torch.ones(n) * i).to(x.device)
#             next_t = (torch.ones(n) * j).to(x.device)
#             at = compute_alpha(b, t.long())
#             at_next = compute_alpha(b, next_t.long())
#             xt = xs[-1].to('cuda')
#
#             if quant:
#                 time_val = len(xs) - 1
#                 model.set_time(time_val)
#
#             if cur_i in interval_seq:
#                 print(f"时间步 {cur_i} 在 interval_seq 中，生成新缓存")
#
#                 current_depth = 0
#                 if cache_sequence is not None:
#                     cache_idx = interval_seq.index(cur_i)
#                     if cache_idx < len(cache_sequence):
#                         current_depth = cache_sequence[cache_idx]
#                         timing_data['current_cache_depth'] = current_depth
#                         print(f"当前缓存深度: {current_depth}")
#
#
#                 if hasattr(model, 'set_random_cache_para') and kwargs.get('args') is not None and kwargs['args'].random_cache:
#                     if cache_sequence is not None:
#                         cache_idx = interval_seq.index(cur_i)
#                         if cache_idx < len(cache_sequence):
#                             cache_index = cache_sequence[cache_idx]
#                             print(f"使用预定义缓存索引: {cache_index}")
#                             model.set_random_cache_para(cache_index=cache_index)
#                     else:
#                         print("不使用预定义缓存序列")
#                         model.set_random_cache_para()
#
#                 # === 准确的GPU时间测量 ===
#                 torch.cuda.synchronize()  # 等待之前的GPU操作完成
#                 start_event = torch.cuda.Event(enable_timing=True)
#                 end_event = torch.cuda.Event(enable_timing=True)
#
#                 start_event.record()  # 记录开始时间
#
#                 et, cur_f = model(xt, t, context=None, prv_f=None, branch=branch)
#
#                 end_event.record()  # 记录结束时间
#                 torch.cuda.synchronize()  # 等待GPU完成
#
#                 elapsed_time_ms = start_event.elapsed_time(end_event)  # 精确的GPU时间
#                 # ========================
#
#                 timing_data['generate_cache_times'].append({
#                     'step': cur_i,
#                     'time_ms': elapsed_time_ms,
#                     'depth': current_depth,
#                     'type': 'generate_cache'
#                 })
#                 print(f"生成缓存调用时间: {elapsed_time_ms:.3f}ms, 深度: {current_depth}")
#
#                 if cur_f and len(cur_f) > 0 and cur_f[0] is not None:
#                     prv_f = cur_f[0]
#                     print(f"✅ 设置prv_f: {prv_f.shape}")
#
#             else:
#                 print(f"时间步 {cur_i} 不在 interval_seq 中，使用缓存")
#
#                 # === 准确的GPU时间测量 ===
#                 torch.cuda.synchronize()  # 等待之前的GPU操作完成
#                 start_event = torch.cuda.Event(enable_timing=True)
#                 end_event = torch.cuda.Event(enable_timing=True)
#
#                 start_event.record()  # 记录开始时间
#
#                 et, cur_f = model(xt, t, context=None, prv_f=prv_f, branch=branch)
#
#                 end_event.record()  # 记录结束时间
#                 torch.cuda.synchronize()  # 等待GPU完成
#
#                 elapsed_time_ms = start_event.elapsed_time(end_event)  # 精确的GPU时间
#                 # ========================
#
#                 current_depth = timing_data['current_cache_depth']
#                 timing_data['use_cache_times'].append({
#                     'step': cur_i,
#                     'time_ms': elapsed_time_ms,
#                     'depth': current_depth,
#                     'type': 'use_cache'
#                 })
#                 print(f"使用缓存调用时间: {elapsed_time_ms:.3f}ms, 深度: {current_depth}")
#
#             timing_data['step_cache_depths'].append({
#                 'step': cur_i,
#                 'depth': timing_data['current_cache_depth']
#             })
#
#             x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#             x0_preds.append(x0_t.to('cpu'))
#             c1 = (
#                     kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
#             )
#             c2 = ((1 - at_next) - c1 ** 2).sqrt()
#             xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
#             xs.append(xt_next.to('cpu'))
#             cur_i += 1
#
#     # 打印并保存统计到.pth文件
#     print_time_statistics_ms(
#         timing_data,
#         save_to_pth=True,
#         pth_dir="./time/",
#         cache_seq_id=None
#     )
#
#     # 将时间数据保存到返回值中
#     if 'timing_results' not in kwargs:
#         kwargs['timing_results'] = []
#     kwargs['timing_results'].append(timing_data)
#
#     return xs, x0_preds


def print_time_statistics_ms(timing_data, save_to_pth=True, pth_dir="./time/", cache_seq_id=None):
    """打印时间统计信息（毫秒版本），可选保存到.pth文件"""

    # 生成统计文本
    stats_lines = []

    stats_lines.append("\n" + "=" * 60)
    stats_lines.append("模型调用时间统计（单位：毫秒）")
    stats_lines.append("=" * 60)

    # 1. 生成缓存的时间统计
    generate_times = timing_data['generate_cache_times']
    if generate_times:
        gen_times_list_ms = [item['time_ms'] for item in generate_times]

        stats_lines.append(f"生成缓存调用次数: {len(generate_times)}")
        stats_lines.append(f"生成缓存总时间: {sum(gen_times_list_ms):.6f}ms")
        stats_lines.append(f"生成缓存平均时间: {np.mean(gen_times_list_ms):.6f}ms")
        stats_lines.append(f"生成缓存最长时间: {max(gen_times_list_ms):.6f}ms")
        stats_lines.append(f"生成缓存最短时间: {min(gen_times_list_ms):.6f}ms")

        # 按深度统计
        depth_stats = {}
        for item in generate_times:
            depth = item['depth']
            if depth not in depth_stats:
                depth_stats[depth] = {'times_ms': [], 'count': 0}
            depth_stats[depth]['times_ms'].append(item['time_ms'])
            depth_stats[depth]['count'] += 1

        stats_lines.append("\n生成缓存按深度统计:")
        for depth in sorted(depth_stats.keys()):
            times_ms = depth_stats[depth]['times_ms']
            stats_lines.append(f"  深度 {depth}: {depth_stats[depth]['count']:3d}次, "
                               f"平均={np.mean(times_ms):10.6f}ms, "
                               f"总计={sum(times_ms):12.6f}ms")

    # 2. 使用缓存的时间统计
    use_times = timing_data['use_cache_times']
    if use_times:
        use_times_list_ms = [item['time_ms'] for item in use_times]

        stats_lines.append(f"\n使用缓存调用次数: {len(use_times)}")
        stats_lines.append(f"使用缓存总时间: {sum(use_times_list_ms):.6f}ms")
        stats_lines.append(f"使用缓存平均时间: {np.mean(use_times_list_ms):.6f}ms")
        stats_lines.append(f"使用缓存最长时间: {max(use_times_list_ms):.6f}ms")
        stats_lines.append(f"使用缓存最短时间: {min(use_times_list_ms):.6f}ms")

        # 按深度统计
        depth_stats = {}
        for item in use_times:
            depth = item['depth']
            if depth not in depth_stats:
                depth_stats[depth] = {'times_ms': [], 'count': 0}
            depth_stats[depth]['times_ms'].append(item['time_ms'])
            depth_stats[depth]['count'] += 1

        stats_lines.append("\n使用缓存按深度统计:")
        for depth in sorted(depth_stats.keys()):
            times_ms = depth_stats[depth]['times_ms']
            stats_lines.append(f"  深度 {depth}: {depth_stats[depth]['count']:3d}次, "
                               f"平均={np.mean(times_ms):10.6f}ms, "
                               f"总计={sum(times_ms):12.6f}ms")

    # 3. 总体统计
    total_calls = len(generate_times) + len(use_times)
    if generate_times and use_times:
        total_time_ms = sum(gen_times_list_ms) + sum(use_times_list_ms)
    elif generate_times:
        total_time_ms = sum(gen_times_list_ms)
    elif use_times:
        total_time_ms = sum(use_times_list_ms)
    else:
        total_time_ms = 0

    stats_lines.append(f"\n总体统计:")
    stats_lines.append(f"总调用次数: {total_calls}次")
    stats_lines.append(f"总调用时间: {total_time_ms:.6f}ms ({total_time_ms / 1000:.6f}s)")
    if total_calls > 0:
        stats_lines.append(f"平均每次调用: {(total_time_ms / total_calls):.6f}ms")
    else:
        stats_lines.append(f"平均每次调用: 无数据")

    # 4. 缓存效率分析
    if generate_times and use_times:
        avg_gen_time_ms = np.mean(gen_times_list_ms)
        avg_use_time_ms = np.mean(use_times_list_ms)

        if avg_gen_time_ms > 0:
            speedup = avg_gen_time_ms / avg_use_time_ms
            time_saved_ms = (avg_gen_time_ms - avg_use_time_ms) * len(use_times)
            stats_lines.append(f"\n缓存效率分析:")
            stats_lines.append(f"  生成缓存平均时间: {avg_gen_time_ms:.6f}ms")
            stats_lines.append(f"  使用缓存平均时间: {avg_use_time_ms:.6f}ms")
            stats_lines.append(f"  缓存加速比: {speedup:.6f}x")
            stats_lines.append(f"  缓存节省总时间: {time_saved_ms:.6f}ms ({time_saved_ms / 1000:.6f}s)")

    stats_lines.append("=" * 60)

    # 将列表转换为字符串
    stats_text = "\n".join(stats_lines)

    # 打印到控制台
    print(stats_text)

    # 保存到.pth文件
    if save_to_pth:
        import os
        import torch
        import datetime

        # 创建目录
        os.makedirs(pth_dir, exist_ok=True)

        # 确定文件名
        # 如果没有指定ID，使用时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pth_filename = os.path.join(pth_dir, f"timing_{timestamp}.pth")

        try:
            # 准备要保存的数据
            save_data = {
                'timing_data': timing_data,  # 原始时间数据
                'stats_text': stats_text,  # 统计文本
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 如果有缓存序列，也保存
            if 'cache_sequence' in timing_data:
                save_data['cache_sequence'] = timing_data['cache_sequence']

            # 保存到.pth文件
            torch.save(save_data, pth_filename)

            print(f"\n✅ 统计结果已保存到: {pth_filename}")

        except Exception as e:
            print(f"❌ 保存.pth文件失败: {e}")

    return stats_text
def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds


def dataset_generate_steps(x, seq, model, b, timesteps, cache_sequence=None, **kwargs):
    print(f"=== dataset_generate_steps: cache_sequence长度={len(cache_sequence) if cache_sequence else 0} ===")

    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        prv_f = None
        cur_i = 0

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            # 检查是否需要处理缓存
            if cache_sequence and cur_i < len(cache_sequence):
                cache_cmd = cache_sequence[cur_i]

                if cache_cmd > 0:
                    # 设置新缓存位置并计算
                    if hasattr(model, 'set_random_cache_para'):
                        model.set_random_cache_para(cache_index=cache_cmd - 1)
                    et, cur_f = model(xt, t, context=None, prv_f=None, branch=None)

                    # 保存缓存特征
                    if cur_f and cur_f[0] is not None:
                        prv_f = cur_f[0]
                else:
                    # 复用缓存
                    et, cur_f = model(xt, t, context=None, prv_f=prv_f, branch=None)
            else:
                # 默认情况
                et, cur_f = model(xt, t, context=None, prv_f=prv_f, branch=None)

            # DDIM更新
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
            cur_i += 1

    return xs, x0_preds