import random
import numpy as np


def get_dirichlet(alpha, n):
    alphas = [alpha] * n
    samples = np.random.dirichlet(alphas, 1).squeeze(0)
    return samples


def get_multinomial(num_samples, prob_vector):
    ms = np.random.choice(len(prob_vector), num_samples, p=prob_vector) + 1
    return ms


def randomly_set_cache_reuse(ms, target_non_zero=13):
    """
    精确控制非零位置数量：
    - 索引0必须是非零（已保证）
    - 索引99必须是0
    - 从索引1-98中随机选择9个设为非零
    - 其他所有位置设为0
    """
    total_steps = len(ms)  # 应该是250

    # 确保索引0是非零（已经在# 从索引1-248中随机选择中保证）
    # ms[0] 已经是1-4

    # # 2. 索引249固定为0（对于250长度）
    ms[249] = 0

    # 计算还需要设置的非零数量（不包括索引0）
    remaining_non_zero = target_non_zero - 1  # 还需要12个

    # 从索引1-248中随机选择
    available_indices = list(range(1, 249))  # [1, 2, 3, ..., 98]

    # 随机选择要设为非零的位置（确保不重复）
    non_zero_indices = random.sample(available_indices, remaining_non_zero)

    # 将所有其他位置（除了索引0和选中的9个）设为0
    for i in range(1, 249):  # 索引1-248
        if i not in non_zero_indices:
            ms[i] = 0

    return ms

def get_non_zero_indices(ms):
    """
    获取列表中所有非零值的索引位置
    返回格式：[索引1, 索引2, ...]
    """
    return [i for i, value in enumerate(ms) if value != 0]

def get_non_zero_values(ms):
    """
    获取列表中所有非零值的序列
    返回格式：[值1, 值2, ...]
    """
    return [value for value in ms if value != 0]

def get_non_zero_values_minus_one(ms):
    """
    获取列表中所有非零值的序列，并将每个值减1
    返回格式：[值1-1, 值2-1, ...]
    """
    return [value - 1 for value in ms if value != 0]


def transform_ms(ms):
    """
    转换ms序列：
    1. 所有非零值变成0
    2. 所有0变成距离自己前面最近的非零值
    """
    n = len(ms)
    result = [0] * n
    last_non_zero = None  # 记录最近的非零值

    for i in range(n):
        if ms[i] != 0:
            # 非零值变成0
            result[i] = 0
            # 更新最近的非零值
            last_non_zero = ms[i]
        else:
            # 0变成距离前面最近的非零值
            if last_non_zero is not None:
                result[i] = last_non_zero
            else:
                # 如果前面没有非零值，保持为0
                result[i] = 0

    return result

# 等价于Z:\OMS-DPM\code\diffusion\examples\ddpm_and_guided-diffusion\utils\get_model_schedule
def get_cache_schedule(config, schedule_length=250, num_cache_positions=None):
    if config.type == "specify":
        pass
    elif config.type == "multinomial":
        prob_vector = config.multinomial.prob_vector
        if len(prob_vector) != num_cache_positions:
            raise ValueError(f"Probability vector length {len(prob_vector)} != cache positions {num_cache_positions}!")

        # 先生成完整的非零序列
        ms = get_multinomial(schedule_length, prob_vector)

        # 精确控制非零数量：第一个固定，再随机选12个
        target_non_zero = 13  # 总共13个非零位置
        randomly_set_cache_reuse(ms, target_non_zero)
        ms = ms.tolist()
    elif config.type == "multinomial+hierarchical":
        alpha = config.hierarchical.alpha
        prob_vector = get_dirichlet(alpha, num_cache_positions)

        # 先生成完整的非零序列
        ms = get_multinomial(schedule_length, prob_vector)
        # 精确控制非零数量：第一个固定，再随机选12个
        target_non_zero = 13
        randomly_set_cache_reuse(ms, target_non_zero)
        ms = ms.tolist()
    else:
        raise NotImplementedError(f"Cache schedule type \"{config.type}\" not supported!")
    non_zero_indices = get_non_zero_indices(ms)
    # 获取非零值的序列
    non_zero_values = get_non_zero_values(ms)
    non_zero_values_minus_one  = get_non_zero_values_minus_one(ms)
    # 进行转换
    transformed_ms = transform_ms(ms)
    print(f"Generated full cache schedule: {transformed_ms}")
    return non_zero_indices,transformed_ms,non_zero_values,non_zero_values_minus_one