import random
import numpy as np


def get_dirichlet(alpha, n):
    alphas = [alpha] * n
    samples = np.random.dirichlet(alphas, 1).squeeze(0)
    return samples


def get_multinomial(num_samples, prob_vector):
    ms = np.random.choice(len(prob_vector), num_samples, p=prob_vector) + 1
    return ms


def randomly_set_cache_reuse(ms, target_non_zero=10):
    """
    精确控制非零位置数量：
    - 索引0必须是非零（已保证）
    - 索引99必须是0
    - 从索引1-98中随机选择9个设为非零
    - 其他所有位置设为0
    """
    total_steps = len(ms)  # 应该是100

    # 确保索引0是非零（已经在get_multinomial中保证）
    # ms[0] 已经是1-4

    # 索引99必须设为0
    ms[99] = 0

    # 计算还需要设置的非零数量（不包括索引0）
    remaining_non_zero = target_non_zero - 1  # 还需要9个

    # 从索引1-98中随机选择（共98个位置）
    available_indices = list(range(1, 99))  # [1, 2, 3, ..., 98]

    # 随机选择要设为非零的位置（确保不重复）
    non_zero_indices = random.sample(available_indices, remaining_non_zero)

    # 将所有其他位置（除了索引0和选中的9个）设为0
    for i in range(1, 99):  # 索引1-98
        if i not in non_zero_indices:
            ms[i] = 0

    return ms

# 等价于Z:\OMS-DPM\code\diffusion\examples\ddpm_and_guided-diffusion\utils\get_model_schedule
def get_cache_schedule(config, schedule_length=100, num_cache_positions=None):
    if config.type == "specify":
        return config.specify.ms
    elif config.type == "multinomial":
        prob_vector = config.multinomial.prob_vector
        if len(prob_vector) != num_cache_positions:
            raise ValueError(f"Probability vector length {len(prob_vector)} != cache positions {num_cache_positions}!")

        # 先生成完整的非零序列
        ms = get_multinomial(schedule_length, prob_vector)

        # 精确控制非零数量：第一个固定，再随机选9个
        target_non_zero = 10  # 总共10个非零位置
        randomly_set_cache_reuse(ms, target_non_zero)

        ms = ms.tolist()
    elif config.type == "multinomial+hierarchical":
        alpha = config.hierarchical.alpha
        prob_vector = get_dirichlet(alpha, num_cache_positions)

        # 先生成完整的非零序列
        ms = get_multinomial(schedule_length, prob_vector)

        # 精确控制非零数量：第一个固定，再随机选9个
        target_non_zero = 10
        randomly_set_cache_reuse(ms, target_non_zero)

        ms = ms.tolist()
    else:
        raise NotImplementedError(f"Cache schedule type \"{config.type}\" not supported!")

    print(f"Generated cache schedule: {ms}")
    return ms