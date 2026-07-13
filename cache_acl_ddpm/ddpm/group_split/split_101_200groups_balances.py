import numpy as np
import random

# 设置随机种子保证可重复性
np.random.seed(42)
random.seed(42)


def generate_sorted_schemes():
    schemes = []

    # 方案1-20: 均匀分布
    for i in range(20):
        step = random.randint(8, 15)
        scheme = [0]
        for j in range(1, 10):
            next_val = min(scheme[-1] + step, 99)  # 确保最后一个数小于100
            scheme.append(next_val)
        # 调整最后一个数确保小于100
        while scheme[-1] >= 100:
            scheme[-1] = random.randint(scheme[-2] + 1, 99)
        schemes.append(scheme)

    # 方案21-40: 前密后疏
    for i in range(20):
        scheme = [0]
        # 前5个较密集
        for j in range(4):
            gap = random.randint(3, 8)
            next_val = min(scheme[-1] + gap, 99)
            scheme.append(next_val)
        # 后5个较稀疏
        for j in range(5):
            gap = random.randint(10, 20)
            next_val = min(scheme[-1] + gap, 99)
            scheme.append(next_val)
        # 确保最后一个数小于100
        while scheme[-1] >= 100:
            scheme[-1] = random.randint(scheme[-2] + 1, 99)
        schemes.append(scheme[:10])  # 确保只有10个数

    # 方案41-60: 前疏后密
    for i in range(20):
        scheme = [0]
        # 前5个较稀疏
        for j in range(4):
            gap = random.randint(10, 20)
            next_val = min(scheme[-1] + gap, 99)
            scheme.append(next_val)
        # 后5个较密集
        for j in range(5):
            gap = random.randint(3, 8)
            next_val = min(scheme[-1] + gap, 99)
            scheme.append(next_val)
        # 确保最后一个数小于100
        while scheme[-1] >= 100:
            scheme[-1] = random.randint(scheme[-2] + 1, 99)
        schemes.append(scheme[:10])

    # 方案61-80: 随机间隔
    for i in range(20):
        scheme = [0]
        current = 0
        for j in range(9):
            gap = random.randint(5, 15)
            current = min(current + gap, 99)
            scheme.append(current)
        # 确保最后一个数小于100
        while scheme[-1] >= 100:
            scheme[-1] = random.randint(scheme[-2] + 1, 99)
        schemes.append(scheme)

    # 方案81-100: 指数增长
    for i in range(20):
        scheme = [0]
        current = 0
        for j in range(1, 10):
            # 指数增长的间隔
            base_gap = random.randint(3, 8)
            gap = min(int(base_gap * (1.1 ** j)), 20)  # 限制最大间隔
            current = min(current + gap, 99)
            scheme.append(current)
        # 确保最后一个数小于100
        while scheme[-1] >= 100:
            scheme[-1] = random.randint(scheme[-2] + 1, 99)
        schemes.append(scheme)

    return schemes


# 生成所有方案
all_schemes = generate_sorted_schemes()

# 输出所有100种方案
print("100种时间步分组方案 (第一个数为0，最后一个数小于100):")
print("=" * 70)

for i, scheme in enumerate(all_schemes, 1):
    print(f"方案{i:2d}: {scheme}")

    # 验证每个方案
    assert len(scheme) == 10, f"方案{i}长度不为10"
    assert scheme[0] == 0, f"方案{i}第一个数不是0"
    assert scheme[-1] < 100, f"方案{i}最后一个数不小于100"
    assert all(0 <= x <= 99 for x in scheme), f"方案{i}有超出范围的值"
    assert scheme == sorted(scheme), f"方案{i}不是递增序列"
    assert len(set(scheme)) == 10, f"方案{i}有重复值"

print("\n所有方案验证通过！")

# 统计信息
print("\n统计信息:")
all_points = [point for scheme in all_schemes for point in scheme]
print(f"总点数: {len(all_points)}")
print(f"覆盖范围: 0-99")
print(f"每个方案点数: 10")
print(f"方案总数: 100")
print(f"所有方案第一个数: 0")
print(f"所有方案最后一个数范围: {min([s[-1] for s in all_schemes])}-{max([s[-1] for s in all_schemes])}")

# 显示点分布密度
distribution = [0] * 10  # 每10个区间
for point in all_points:
    distribution[point // 10] += 1

print("\n点分布密度 (每10个区间的点数):")
for i in range(10):
    start = i * 10
    end = start + 9
    print(f"区间[{start:2d}-{end:2d}]: {distribution[i]:4d}个点")

# 显示最后一个数的分布
last_numbers = [scheme[-1] for scheme in all_schemes]
print(f"\n最后一个数的分布:")
for i in range(10):
    start = i * 10
    end = start + 9
    count = len([x for x in last_numbers if start <= x <= end])
    print(f"区间[{start:2d}-{end:2d}]: {count:2d}个方案")