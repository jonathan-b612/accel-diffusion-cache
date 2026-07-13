import random
import numpy as np
from typing import List, Dict
import math


def generate_truly_diverse_groupings(total_steps: int = 100,
                                     num_groups: int = 10,
                                     total_samples: int = 100) -> List[List[int]]:
    """
    生成真正多样化的分组方案，包含极端情况
    """
    groupings = []

    # 为每个策略生成更多方案以确保去重后有足够数量
    samples_per_strategy = total_samples // 5  # 每个策略生成20个

    # 1. 极端前期密集 (第一步很小)
    groupings.extend(generate_extreme_early_dense(total_steps, num_groups, samples_per_strategy))

    # 2. 极端后期密集 (第一步很大)
    groupings.extend(generate_extreme_late_dense(total_steps, num_groups, samples_per_strategy))

    # 3. 极端均匀分布
    groupings.extend(generate_extreme_uniform(total_steps, num_groups, samples_per_strategy))

    # 4. 双峰分布 (两个密集区域)
    groupings.extend(generate_bimodal_groupings(total_steps, num_groups, samples_per_strategy))

    # 5. 随机但包含极端值
    groupings.extend(generate_extreme_random(total_steps, num_groups, samples_per_strategy))

    print(f"去重前生成 {len(groupings)} 个方案")

    # 去重
    unique_groupings = remove_duplicates(groupings)
    print(f"去重后剩余 {len(unique_groupings)} 个方案")

    # 如果数量不足，补充更多极端方案
    if len(unique_groupings) < total_samples:
        additional_needed = total_samples - len(unique_groupings)
        print(f"需要补充 {additional_needed} 个方案")
        additional = generate_more_extreme_variants(total_steps, num_groups, additional_needed * 2)
        additional_unique = remove_duplicates(additional)
        unique_groupings.extend(additional_unique[:additional_needed])

    # 最终确保数量
    final_groupings = unique_groupings[:total_samples]
    print(f"最终生成 {len(final_groupings)} 个方案")

    return final_groupings


def generate_extreme_early_dense(total_steps: int, num_groups: int, num_samples: int) -> List[List[int]]:
    """极端前期密集 - 第一步非常小"""
    groupings = []

    for i in range(num_samples):
        # 第一步可以小到1-5
        first_step = random.randint(1, 5)

        # 生成指数衰减的间隔
        decay_factor = random.uniform(1.5, 3.0)
        base_intervals = [first_step * (decay_factor ** j) for j in range(num_groups)]

        # 调整总和
        total = sum(base_intervals)
        if total > 0:
            scale = total_steps / total
            intervals = [max(1, int(x * scale)) for x in base_intervals]
            adjust_sum(intervals, total_steps)

            grouping = accumulate_intervals(intervals)
            groupings.append(grouping)

    return groupings


def generate_extreme_late_dense(total_steps: int, num_groups: int, num_samples: int) -> List[List[int]]:
    """极端后期密集 - 第一步非常大"""
    groupings = []

    for i in range(num_samples):
        # 第一步可以大到20-40
        first_step = random.randint(20, 40)

        # 生成递减的间隔
        decay_factor = random.uniform(0.4, 0.8)
        base_intervals = [first_step * (decay_factor ** j) for j in range(num_groups)]

        # 调整总和
        total = sum(base_intervals)
        if total > 0:
            scale = total_steps / total
            intervals = [max(1, int(x * scale)) for x in base_intervals]
            adjust_sum(intervals, total_steps)

            grouping = accumulate_intervals(intervals)
            groupings.append(grouping)

    return groupings


def generate_extreme_uniform(total_steps: int, num_groups: int, num_samples: int) -> List[List[int]]:
    """极端均匀分布"""
    groupings = []

    base_interval = total_steps // num_groups

    for i in range(num_samples):
        grouping = [0]
        current = 0

        # 几乎完美的均匀分布
        if random.random() < 0.3:
            # 完全均匀
            for j in range(num_groups - 1):
                current += base_interval
                grouping.append(current)
        else:
            # 轻微扰动但保持高度均匀
            for j in range(num_groups - 1):
                perturbation = random.randint(-2, 2)
                interval = max(1, base_interval + perturbation)
                current = min(current + interval, total_steps)
                grouping.append(current)

        grouping[-1] = total_steps  # 确保最后是100
        groupings.append(grouping)

    return groupings


def generate_bimodal_groupings(total_steps: int, num_groups: int, num_samples: int) -> List[List[int]]:
    """双峰分布 - 两个密集区域"""
    groupings = []

    for i in range(num_samples):
        # 选择两个密集中心
        center1 = random.uniform(0.1, 0.4)
        center2 = random.uniform(0.6, 0.9)
        std = random.uniform(0.05, 0.15)

        positions = np.linspace(0, 1, num_groups)
        weights = [
            math.exp(-0.5 * ((pos - center1) / std) ** 2) +
            math.exp(-0.5 * ((pos - center2) / std) ** 2)
            for pos in positions
        ]

        # 转换为间隔
        total_weight = sum(weights)
        if total_weight > 0:
            intervals = [max(1, int((w / total_weight) * total_steps)) for w in weights]
            adjust_sum(intervals, total_steps)

            grouping = accumulate_intervals(intervals)
            groupings.append(grouping)

    return groupings


def generate_extreme_random(total_steps: int, num_groups: int, num_samples: int) -> List[List[int]]:
    """包含极端值的随机分布"""
    groupings = []

    for i in range(num_samples):
        intervals = []

        for j in range(num_groups):
            # 有概率生成极端值
            if random.random() < 0.4:  # 40%概率生成极端值
                if random.random() < 0.5:
                    # 极小值
                    intervals.append(random.randint(1, 3))
                else:
                    # 极大值
                    intervals.append(random.randint(20, 35))
            else:
                # 正常值
                intervals.append(random.randint(5, 15))

        # 调整总和
        adjust_sum(intervals, total_steps)
        grouping = accumulate_intervals(intervals)
        groupings.append(grouping)

    return groupings


def generate_more_extreme_variants(total_steps: int, num_groups: int, num_samples: int) -> List[List[int]]:
    """生成更多极端变体"""
    groupings = []

    for i in range(num_samples):
        # 完全随机但强制包含极端值
        intervals = []

        # 确保至少有一个极小值和一个极大值
        intervals.append(random.randint(1, 3))  # 极小值
        intervals.append(random.randint(25, 40))  # 极大值

        # 填充其余
        for j in range(num_groups - 2):
            # 随机选择：小值、正常值或大值
            choice = random.random()
            if choice < 0.3:
                intervals.append(random.randint(1, 4))
            elif choice < 0.6:
                intervals.append(random.randint(15, 30))
            else:
                intervals.append(random.randint(5, 12))

        adjust_sum(intervals, total_steps)
        grouping = accumulate_intervals(intervals)
        groupings.append(grouping)

    return groupings


def accumulate_intervals(intervals: List[int]) -> List[int]:
    """将间隔累积为分组点"""
    grouping = [0]
    current = 0
    for interval in intervals:
        current = min(current + interval, 100)
        grouping.append(current)
    grouping[-1] = 100  # 确保最后是100
    return grouping


def adjust_sum(intervals: List[int], target_sum: int):
    """调整间隔列表的总和为目标值"""
    current_sum = sum(intervals)
    diff = target_sum - current_sum

    if diff != 0:
        # 在最大的间隔上调整（避免影响小间隔）
        max_idx = intervals.index(max(intervals))
        intervals[max_idx] += diff
        intervals[max_idx] = max(1, intervals[max_idx])


def remove_duplicates(groupings: List[List[int]]) -> List[List[int]]:
    """去除重复的分组方案"""
    seen = set()
    unique = []

    for grouping in groupings:
        grouping_tuple = tuple(grouping)
        if grouping_tuple not in seen:
            seen.add(grouping_tuple)
            unique.append(grouping)

    return unique


def analyze_first_step_distribution(groupings: List[List[int]]):
    """分析第一步的分布"""
    first_steps = []
    for grouping in groupings:
        first_step = grouping[1] - grouping[0]  # 第一步的大小
        first_steps.append(first_step)

    print(f"\n第一步分布分析:")
    print(f"最小值: {min(first_steps)}")
    print(f"最大值: {max(first_steps)}")
    print(f"平均值: {np.mean(first_steps):.2f}")
    print(f"标准差: {np.std(first_steps):.2f}")

    # 统计不同范围的分布
    ranges = [(1, 3), (4, 8), (9, 15), (16, 25), (26, 100)]
    for r_min, r_max in ranges:
        count = sum(1 for step in first_steps if r_min <= step <= r_max)
        print(f"第一步在 {r_min}-{r_max}: {count} 个方案 ({count / len(first_steps) * 100:.1f}%)")


# 使用示例
if __name__ == "__main__":
    # 生成真正多样化的分组方案
    diverse_groupings = generate_truly_diverse_groupings(
        total_steps=100,
        num_groups=10,
        total_samples=100
    )

    print(f"成功生成 {len(diverse_groupings)} 种分组方案")

    # 分析第一步分布
    analyze_first_step_distribution(diverse_groupings)

    # 打印所有方案
    print(f"\n{'=' * 60}")
    print("所有100种分组方案:")
    print(f"{'=' * 60}")

    for i, grouping in enumerate(diverse_groupings):
        intervals = [grouping[j + 1] - grouping[j] for j in range(len(grouping) - 1)]
        print(f"方案 {i + 1:3d}: {grouping} -> 间隔: {intervals}")

    # 保存到Python文件
    with open("diverse_interval_sequences.py", "w", encoding="utf-8") as f:
        f.write('"""\n')
        f.write('100种多样化时间步分组方案\n')
        f.write('总步数: 100, 分组数: 10\n')
        f.write('用于替换 get_interval_seq 函数的返回值\n')
        f.write('"""\n\n')

        f.write('DIVERSE_INTERVAL_SEQUENCES = [\n')
        for i, grouping in enumerate(diverse_groupings):
            intervals = [grouping[j + 1] - grouping[j] for j in range(len(grouping) - 1)]
            f.write(f"    # 方案 {i + 1}: 间隔 {intervals}\n")
            f.write(f"    {grouping},\n")
        f.write(']\n\n')

        f.write('def get_diverse_interval_sequence(index):\n')
        f.write('    """\n')
        f.write('    获取第index种分组方案\n')
        f.write('    参数:\n')
        f.write('        index: 方案索引 (0-99)\n')
        f.write('    返回:\n')
        f.write('        分组序列列表，如 [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]\n')
        f.write('    """\n')
        f.write('    if 0 <= index < len(DIVERSE_INTERVAL_SEQUENCES):\n')
        f.write('        return DIVERSE_INTERVAL_SEQUENCES[index]\n')
        f.write('    else:\n')
        f.write('        raise ValueError(f"索引必须在0-{len(diverse_groupings)-1}之间")\n\n')

        f.write('def get_all_interval_sequences():\n')
        f.write('    """\n')
        f.write('    获取所有100种分组方案\n')
        f.write('    返回:\n')
        f.write('        包含所有分组方案的列表\n')
        f.write('    """\n')
        f.write('    return DIVERSE_INTERVAL_SEQUENCES\n')

    print(f"\n{'=' * 60}")
    print("分组方案已保存到 diverse_interval_sequences.py")
    print(f"{'=' * 60}")