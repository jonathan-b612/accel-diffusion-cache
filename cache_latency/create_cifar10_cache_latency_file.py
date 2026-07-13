import torch
import os

"""
这里存储的是一个批次，一个step所花的时间,batch_size = 500
总共100个批次，100个step
实验里面的budget代表的是一个批次的1000个step所花的时间

Budget/ms	Baseline 能跑的步数
2.5 × 10⁴	125 步
2.0 × 10⁴	100 步
1.5 × 10⁴	75 步
1.0 × 10⁴	50 步
"""
def create_cache_latency_file():
    """创建缓存延迟文件，只包含五个数值"""

    # 您提供的五个数值
    cache_latency_data = [320.18, 314.68, 296.95, 190.23, 36.83]

    print("要保存的数据:")
    print(f"数据: {cache_latency_data}")
    print(f"数据类型: {type(cache_latency_data)}")
    print(f"数据长度: {len(cache_latency_data)}")

    # 确保目录存在
    output_dir = 'cifar10'
    os.makedirs(output_dir, exist_ok=True)

    # 保存文件
    output_path = os.path.join(output_dir, 'cifar10_cache_latency.pth')

    # 使用torch.save保存
    torch.save(cache_latency_data, output_path)

    print(f"\n✅ 已创建文件: {output_path}")
    print(f"保存的数据: {cache_latency_data}")

    # 验证保存的文件
    print("\n验证保存的文件:")
    loaded_data = torch.load(output_path, map_location='cpu')
    print(f"加载的数据类型: {type(loaded_data)}")
    print(f"加载的数据: {loaded_data}")
    print(f"数据长度: {len(loaded_data)}")

    # 详细验证每个值
    print("\n详细验证:")
    for i, (saved_val, loaded_val) in enumerate(zip(cache_latency_data, loaded_data)):
        # 计算相对误差
        if isinstance(saved_val, (int, float)) and isinstance(loaded_val, (int, float)):
            saved_float = float(saved_val)
            loaded_float = float(loaded_val)
            diff = abs(saved_float - loaded_float)
            rel_error = diff / saved_float * 100 if saved_float != 0 else 0

            if diff < 1e-6:
                print(f"✅ 索引[{i}]: {saved_val} == {loaded_val} (完全匹配)")
            elif rel_error < 0.001:  # 误差小于0.001%
                print(f"✅ 索引[{i}]: {saved_val} ≈ {loaded_val} (误差: {rel_error:.6f}%)")
            else:
                print(f"⚠️  索引[{i}]: {saved_val} != {loaded_val} (误差: {rel_error:.6f}%)")
        else:
            print(f"❌ 索引[{i}]: 类型不匹配 {type(saved_val)} vs {type(loaded_val)}")

    # 总结
    if len(cache_latency_data) == len(loaded_data):
        all_match = True
        for i in range(len(cache_latency_data)):
            saved_val = cache_latency_data[i]
            loaded_val = loaded_data[i]

            if isinstance(saved_val, torch.Tensor) and isinstance(loaded_val, torch.Tensor):
                if not torch.allclose(saved_val, loaded_val, rtol=1e-6):
                    all_match = False
                    break
            elif isinstance(saved_val, (int, float)) and isinstance(loaded_val, (int, float)):
                if abs(float(saved_val) - float(loaded_val)) > 1e-6:
                    all_match = False
                    break
            else:
                all_match = False
                break

        if all_match:
            print("\n🎉 所有验证通过！文件保存成功且数据完全一致。")
        else:
            print("\n⚠️  验证警告：存在微小数值差异，但文件格式正确。")
    else:
        print("\n❌ 验证失败：数据长度不一致！")

    return cache_latency_data


if __name__ == "__main__":
    print("=" * 60)
    print("创建 cifar10_cache_latency.pth 文件")
    print("只包含五个数值: [320.18, 314.68, 296.95, 190.23, 36.83]")
    print("=" * 60)

    # 创建文件
    data = create_cache_latency_file()

    print("\n" + "=" * 60)
    print("文件创建完成！")
    print("=" * 60)
    print(f"\n文件位置: {os.path.abspath('cifar10/cifar10_cache_latency.pth')}")
    print(f"包含的五个数值:")
    for i, val in enumerate(data):
        print(f"  索引[{i}]: {val:.6f} ms")