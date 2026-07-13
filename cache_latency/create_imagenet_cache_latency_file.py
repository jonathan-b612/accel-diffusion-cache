import torch
import os

"""
这里存储的是一个批次，一个step所花的时间,batch_size = 250
总共2000个批次，250个step
实验里面的budget代表的是一个批次的1000个step所花的时间

原始预算(ms)   Baseline 能跑的步数
120,000        2,481 步
110,000        2,274 步
100,000        2,067 步
90,000         1,860 步
80,000         1,654 步
70,000         1,447 步
60,000         1,240 步
50,000         1,034 步
40,000         827 步
30,000         620 步
20,000         413 步
17,500         362 步
15,000         310 步
12,500         258 步
10,000         207 步

"""

def create_imagenet_latency_file():
    """创建ImageNet缓存延迟文件，只包含五个数值"""

    # 五个数值
    imagenet_latency_data = [466.30, 414.43, 331.27, 160.42, 25.45]

    # 确保目录存在
    output_dir = 'imagenet'
    os.makedirs(output_dir, exist_ok=True)

    # 保存文件
    output_path = os.path.join(output_dir, 'imagenet_cache_latency.pth')

    # 使用torch.save保存
    torch.save(imagenet_latency_data, output_path)

    return imagenet_latency_data


if __name__ == "__main__":
    # 创建文件
    create_imagenet_latency_file()