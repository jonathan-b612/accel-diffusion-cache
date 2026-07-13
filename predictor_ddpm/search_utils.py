import os
import torch
import random
import numpy as np
import copy

class model_schedule:
    def __init__(self, config):
        # self.model_zoo_size = config["predictor"]["model_embedder"]["model_zoo_size"] + 1 # +1 for the null model
        self.model_zoo_size = config["predictor"]["model_embedder"]["cache_zoo_size"] + 1 # +1 for the null model
        self.max_length = config["search"]["max_length"]
        self.mutate_prob = config["search"]["mutate_prob"]
        self.init_correct_prob = config["search"]["init_correct_prob"]
        # self.step_size = config["search"]["step_size"]
        self.step_size = config["search"]["cache_step_size"]

        # load model zoo latency
        self.model_zoo_latency = torch.load(config["search"]["cache_zoo_latency"])
        # if self.model_zoo_latency[0]!=0: # null model
        #     self.model_zoo_latency.insert(0, 0)
        self.model_zoo_latency = np.array(self.model_zoo_latency)

        self.cost = None
        self.perf = None
        self.ms = None # numpy
        self.cache_index = None

    def set_ms(self, ms):
        self.ms = ms

    def set_cache_index(self, cache_index):
        self.cache_index = cache_index

    def random_init_ms(self):
        pass

    def get_perf(self, predictor):
        data = {
            "ms":torch.from_numpy(self.ms).to(next(predictor.parameters()).device).unsqueeze(0)
        }
        if self.perf is None:
            with torch.no_grad():
                self.perf = predictor(data).item()
        return self.perf

    def get_cost(self):
        if self.cost is None:
            self.cost = np.sum(self.model_zoo_latency[self.ms]).item()
        return self.cost

    # def get_cost_cache(self):
    #     """动态计算成本：0的时间依赖最近缓存模型，未初始化时报错"""
    #     if self.cost is not None:
    #         return self.cost
    #
    #     total_cost = 0
    #     current_cache_model = None  # 当前缓存是哪个模型的结果
    #
    #     # 从 model_zoo_latency 获取时间数据
    #     # 假设：model_zoo_latency[0] = 完整计算时间
    #     #       model_zoo_latency[1-4] = 各个模型的缓存读取时间
    #
    #     for step, strategy in enumerate(self.ms):
    #         if strategy == 0:  # 使用缓存
    #             if current_cache_model is not None:
    #                 # 读取当前缓存模型对应的缓存读取时间
    #                 # current_cache_model 是 1-4 之间的数字
    #                 if 1 <= current_cache_model <= 4:
    #                     cache_read_time = self.model_zoo_latency[current_cache_model]
    #                     total_cost += cache_read_time
    #                 else:
    #                     # 如果 current_cache_model 不是 1-4，可能是其他模型
    #                     # 这里可以根据需要处理，比如使用默认值或报错
    #                     raise ValueError(f"缓存模型索引 {current_cache_model} 无效！")
    #             else:
    #                 # 缓存未初始化，直接报错
    #                 raise ValueError(f"在第{step}步尝试使用缓存，但缓存未初始化！")
    #         else:  # 策略1-4：完整计算（时间相同）
    #             # 使用 model_zoo_latency[0] 作为完整计算时间
    #             compute_time = self.model_zoo_latency[0]
    #             total_cost += compute_time
    #
    #             # 计算后更新缓存为当前模型
    #             current_cache_model = strategy
    #
    #     self.cost = total_cost
    #     return self.cost
    def mutate(self):
        '''
        Return a new model schedule instance while keeping the current model schedule instance unchanged
        '''
        pass

    def correct_cost(self, tolerance, max_init_time, budget):
        init_time = 0
        while(1):
            print("================================",self.get_cost())
            if self.get_cost()>budget:
                self.mutate(mode="lighter", return_type="self", mutate_prob=self.init_correct_prob)
            elif self.get_cost()<tolerance * budget:
                self.mutate(mode="heavier", return_type="self", mutate_prob=self.init_correct_prob)
            else:
                print(f"Finish correcting.")
                break
            init_time += 1
            if init_time>max_init_time:
                print(f"Faile to initialize a model schedule whose cost is in [{tolerance}*{budget}, {budget}]. The cost of initial model schedule is {self.get_cost()}")
                import ipdb; ipdb.set_trace()
                break

    def check(self):
        '''
        Check whether the format of model schedule is correct
        '''
        pass

class dpmsolver_model_schedule(model_schedule):
    def __init__(self, config):
        super().__init__(config)
        
    def random_init_ms(self, tolerance=0.9, max_init_time=200, budget=None):
        ms = []
        K = self.max_length // 3
        for i in range(K):
            solver_order = random.randint(0, 3)
            for j in range(0, 3):
                if j<solver_order:
                    ms.append(random.randint(1, self.model_zoo_size-1))
                else:
                    ms.append(0)
        ms = np.array(ms)
        self.set_ms(ms)
        
        if budget is not None:
            self.correct_cost(tolerance, max_init_time, budget)
        print(f"Finish initialization. The cost of initial model schedule is {self.get_cost()}")
            
    def mutate(self, mode="normal", return_type="new", mutate_prob=None):
        assert mode in ["normal", "heavier", "lighter"]
        assert return_type in ["new", "self"]
        
        if mutate_prob is None:
            mutate_prob = self.mutate_prob
        
        if return_type=="new":
            new_ms = copy.deepcopy(self)
            prev_ms = self.ms
        else:
            new_ms = self
            prev_ms = copy.deepcopy(self.ms)
            
        random_step_size = random.randint(1, self.step_size)
        indices = torch.tensor(random.sample(range(0, len(new_ms.ms)), random_step_size))
        for index in indices:
            if index % 3!=2 and new_ms.ms[index+1]!=0:
                candidate_list = np.array(range(1, self.model_zoo_size))
            else:
                candidate_list = np.array(range(self.model_zoo_size))
            if mode=="heavier":
                candidate_list = candidate_list[np.where(new_ms.model_zoo_latency[candidate_list]>=new_ms.model_zoo_latency[new_ms.ms[index]])[0]]
            elif mode=="lighter":
                candidate_list = candidate_list[np.where(new_ms.model_zoo_latency[candidate_list]<=new_ms.model_zoo_latency[new_ms.ms[index]])[0]]
            if index % 3==0 or new_ms.ms[index-1]!=0:
                if random.uniform(0, 1)<mutate_prob:
                    new_ms.ms[index] = random.choice(candidate_list)
                    
        if not (new_ms.ms==prev_ms).all():
            new_ms.cost = None
            new_ms.perf = None
                    
        if return_type=="new":
            return new_ms
    
    def check(self):
        if len(self.ms)!=self.max_length:
            return False
        for i in range(self.max_length):
            if self.ms[i]==0 and self.ms[i+1]!=0 and i % 3!=2:
                return False
        return True
    
class ddim_model_schedule(model_schedule):
    def __init__(self, config):
        super().__init__(config)
        
    # def random_init_ms(self, tolerance=0.9, max_init_time=200, budget=None):
    #     ms = np.random.randint(low=0, high=self.model_zoo_size, size=self.max_length)
    #     self.set_ms(ms)
    #
    #     if budget is not None:
    #         self.correct_cost(tolerance, max_init_time, budget)
    #     print(f"Finish initialization. The cost of initial model schedule is {self.get_cost()}")

    # def random_init_ms(self, tolerance=0.9, max_init_time=200, budget=None):
    #     # ms = np.random.randint(low=0, high=self.model_zoo_size, size=self.max_length)
    #     # self.set_ms(ms)
    #
    #     if budget is not None:
    #         self.correct_cost(tolerance, max_init_time, budget)
    #     print(f"Finish initialization. The cost of initial model schedule is {self.get_cost()}")
    #     # 原有的随机生成逻辑
    #     # 初始化100个时间步，全部设为0（使用缓存）
    #     full_sequence = [0] * 100
    #
    #     # 生成索引序列 [0, 1, 2, 3] 的随机组合（四个位置）
    #     sequence = [random.randint(0, self.model_zoo_size -1) for _ in range(num_positions)]
    #
    #     # 在interval_seq指定的索引位置设置缓存位置，使用sequence中的值+1（变成1-4）
    #     for i, idx in enumerate(self.interval_seq):
    #         if idx < len(full_sequence) and i < len(sequence):  # 确保索引在范围内
    #             cache_position = sequence[i] + 1  # 0-3 变成 1-4
    #             full_sequence[idx] = cache_position
    #
    #     # 实际使用的位置就是sequence中的值+1
    #     used_positions = [val + 1 for val in sequence]  # 0-3 变成 1-4
    #     self.set_ms(full_sequence)
    #     if budget is not None:
    #         self.correct_cost(tolerance, max_init_time, budget)
    #     print(f"Finish initialization. The cost of initial model schedule is {self.get_cost}")

    # def random_init_ms(self, tolerance=0.9, max_init_time=200, budget=None):
    #     """随机初始化ms序列（保证正好10个缓存点，0是缓存点，99不是缓存点）"""
    #     # 1. 固定规则：
    #     #    - 位置0必须是缓存点
    #     #    - 从1-98中随机选择9个位置作为其他缓存点
    #     #    - 位置99不能是缓存点
    #     available_positions = list(range(1, self.max_length - 1))  # 1-98
    #     additional_indices = sorted(random.sample(available_positions, 9))
    #
    #     # 2. 组合成10个缓存点位置
    #     cache_indices = [0] + additional_indices
    #
    #     # 3. 为每个缓存点随机分配深度（1-4），并构建cache_index列表
    #     depths = []
    #     for i in range(len(cache_indices)):
    #         depth = random.randint(1, 4)
    #         depths.append(depth)
    #
    #     # 4. 构建cache_index：10个元组(位置, 深度)
    #     cache_index = []
    #     for i in range(len(cache_indices)):
    #         cache_index.append((cache_indices[i], depths[i]))
    #
    #     # 5. 直接生成ms序列
    #     ms = np.zeros(self.max_length, dtype=int)
    #     for i in range(len(cache_indices)):
    #         ms[cache_index[i]] = 0
    #         start_pos = cache_indices[i]
    #         depth = depths[i]
    #
    #         # 确定结束位置
    #         if i < len(cache_indices) - 1:
    #             end_pos = cache_indices[i + 1]  # 到下一个缓存点之前
    #         else:
    #             end_pos = self.max_length  # 最后一个到序列末尾
    #
    #         # 填充深度值
    #         for pos in range(start_pos, end_pos):
    #             ms[pos] = depth
    #
    #     # 6. 调用set_ms统一设置
    #     self.set_ms(ms)
    #     self.set_cache_index(cache_index)
    #     if budget is not None:
    #         self.correct_cost(tolerance, max_init_time, budget)
    #     print(f"Finish initialization. The cost of initial model schedule is {self.get_cost()}")

    def random_init_ms(self, tolerance=0.95, max_init_time=200, budget=None):
        """位置0固定为缓存点，1-98随机选9个缓存点"""

        # 1. 确定缓存点位置
        other_cache_positions = random.sample(range(1, 99), 9)
        cache_positions = [0] + sorted(other_cache_positions)
        # 2. 为每个段随机生成深度（1-4）
        depths = [random.randint(1, 4) for _ in range(10)]
        # cache_positions = [i * 10 for i in range(10)] #做消融 固定间隔
        # depths = [1 for _ in range(10)] #做消融 固定深度
        # 3. 生成完整序列（长度100）
        ms = np.zeros(100, dtype=int)

        for i in range(len(cache_positions)):
            start = cache_positions[i]
            depth = depths[i]

            if i < len(cache_positions) - 1:
                end = cache_positions[i + 1]
            else:
                end = 100

            ms[start] = 0
            ms[start + 1:end] = depth

        # 4. 关键：设置cache_index为(位置, 深度)的列表
        self.ms = ms
        self.cache_index = [(cache_positions[i], depths[i]) for i in range(10)]

        print(f"初始化完成: cache_index长度={len(self.cache_index)}")
        print(f"cache_index: {self.cache_index[:]}")

        # 5. 如果需要修正成本
        if budget is not None:
            self.correct_cost(tolerance, max_init_time, budget)

        print(f"Finish initialization. The cost of initial model schedule is {self.get_cost()}")

    # def mutate(self, mode="normal", return_type="new", mutate_prob=None):
    #     assert mode in ["normal", "heavier", "lighter"]
    #     assert return_type in ["new", "self"]
    #
    #     if mutate_prob is None:
    #         mutate_prob = self.mutate_prob
    #
    #     if return_type=="new":
    #         new_ms = copy.deepcopy(self)
    #         prev_ms = self.ms
    #     else:
    #         new_ms = self
    #         prev_ms = copy.deepcopy(self.ms)
    #
    #     random_step_size = random.randint(1, self.step_size)
    #     indices = torch.tensor(random.sample(range(0, len(new_ms.ms)), random_step_size))
    #     for index in indices:
    #         candidate_list = np.array(range(self.model_zoo_size))
    #         if mode=="heavier":
    #             candidate_list = candidate_list[np.where(new_ms.model_zoo_latency[candidate_list]>=new_ms.model_zoo_latency[new_ms.ms[index]])[0]]
    #         elif mode=="lighter":
    #             candidate_list = candidate_list[np.where(new_ms.model_zoo_latency[candidate_list]<=new_ms.model_zoo_latency[new_ms.ms[index]])[0]]
    #         if random.uniform(0, 1)<mutate_prob:
    #             new_ms.ms[index] = random.choice(candidate_list)
    #     if not (new_ms.ms==prev_ms).all():
    #         new_ms.cost = None
    #         new_ms.perf = None
    #
    #     if return_type=="new":
    #         return new_ms
    #

    def mutate(self, mode="normal", return_type="new", mutate_prob=None):
        """变异操作：移动缓存点位置或改变缓存深度"""
        assert mode in ["normal", "heavier", "lighter"]
        assert return_type in ["new", "self"]

        if mutate_prob is None:
            mutate_prob = self.mutate_prob

        if return_type == "new":
            new_ms = copy.deepcopy(self)
            prev_cache_index = copy.deepcopy(self.cache_index)
        else:
            new_ms = self
            prev_cache_index = copy.deepcopy(self.cache_index)

        # 确定要变异的缓存点数量
        random_step_size = random.randint(1, self.step_size)  # 至少变异一个点
        cache_indices_to_mutate = random.sample(range(len(self.cache_index)), random_step_size)


        mutations_made = False
        for cache_idx in cache_indices_to_mutate:
            current_pos, current_depth = new_ms.cache_index[cache_idx]

            # 随机选择变异类型
            mutation_type = random.choice(["move", "change_depth"])
            # mutation_type = random.choice(["move"]) # 这里做消融实验的时候，我们把这里的任意选择的改成只选择一个
            # mutation_type = random.choice(["change_depth"]) # 这里做消融实验的时候，我们把这里的任意选择的改成只选择一个

            if mutation_type == "move":
                # 移动位置
                if random.uniform(0, 1) < mutate_prob:
                    new_pos = new_ms._get_new_position(cache_idx, current_pos, mode)
                    if new_pos is not None and new_pos != current_pos:
                        new_ms.cache_index[cache_idx] = (new_pos, current_depth)
                        mutations_made = True
                    else:
                        pass
                        # print(f"  Move failed (new_pos={new_pos})")

            elif mutation_type == "change_depth":
                # 改变深度
                if random.uniform(0, 1) < mutate_prob:
                    new_depth = new_ms._get_new_depth(current_depth, mode)
                    if new_depth != current_depth:
                        new_ms.cache_index[cache_idx] = (current_pos, new_depth)
                        mutations_made = True
                    else:
                        pass
                        # print(f"  Depth unchanged")

        if mutations_made:
            # 更新ms序列
            new_ms._update_ms_from_cache_index()
            # 验证生成的序列
            # if not new_ms.validate_ms_sequence():
            #     print("⚠️ 警告: 生成的ms序列验证失败")

            new_ms.cost = None
            new_ms.perf = None
        else:
            pass
            # print("\nNo mutations made")

        if return_type == "new":
            return new_ms

    # def _get_move_range(self, cache_point_idx):
    #     """计算缓存点可以向左和向右移动的步数
    #
    #     规则：
    #     1. 两个缓存点之间必须至少有一个0值（完整计算）时间步
    #     2. 第一个缓存点（位置0）不能移动
    #     3. 最后一个缓存点不能移动到位置99
    #
    #     Args:
    #         cache_point_idx: 缓存点在cache_index列表中的索引（0-9）
    #
    #     Returns:
    #         (left_limit, right_limit): 可以向左和向右移动的步数
    #     """
    #     current_pos, current_cache_idx = self.cache_index[cache_point_idx]
    #
    #     print(f"\n计算缓存点[{cache_point_idx}] ({current_pos}, {current_cache_idx})的移动范围:")
    #
    #     # 规则1：第一个缓存点（位置0）不能移动
    #     # 修正：这里应该只检查位置0，而不是索引0
    #     if current_pos == 0:
    #         print("  缓存点在位置0，不能移动")
    #         return 0, 0
    #
    #     # 1. 向左移动的限制
    #     left_limit = 0
    #     if cache_point_idx > 0:
    #         # 左边有缓存点
    #         left_pos, left_cache_idx = self.cache_index[cache_point_idx - 1]
    #
    #         # 可以向左移动的最大步数 = 当前时间步 - 左边时间步 - 1
    #         # 修正：确保至少有1个0值间隔
    #         left_limit = current_pos - left_pos - 1
    #
    #         print(f"  左边缓存点: ({left_pos}, {left_cache_idx})")
    #         print(f"  需要保持至少1个0值间隔: {current_pos} - {left_pos} - 1 = {left_limit}")
    #
    #     # 2. 向右移动的限制
    #     right_limit = 0
    #     if cache_point_idx < len(self.cache_index) - 1:
    #         # 右边有缓存点
    #         right_pos, right_cache_idx = self.cache_index[cache_point_idx + 1]
    #
    #         # 可以向右移动的最大步数 = 右边时间步 - 当前时间步 - 1
    #         # 修正：公式统一
    #         right_limit = right_pos - current_pos - 1
    #
    #         print(f"  右边缓存点: ({right_pos}, {right_cache_idx})")
    #         print(f"  需要保持至少1个0值间隔: {right_pos} - {current_pos} - 1 = {right_limit}")
    #     else:
    #         # 这是最后一个缓存点
    #         # 规则3：不能移动到位置99
    #         # 修正：最后一个缓存点可以移动到98，但99会被覆盖
    #
    #         # 如果当前在位置96：
    #         # - 移动到97: 可以（97是缓存点，98-99被覆盖）
    #         # - 移动到98: 可以（98是缓存点，99被覆盖）
    #         # - 移动到99: 不可以（99是缓存点，没有位置被覆盖）
    #
    #         max_right = self.max_length - current_pos - 2
    #         right_limit = max(0, max_right)
    #
    #         print(f"  最后一个缓存点，限制向右移动不超过位置{self.max_length - 2}")
    #         print(f"  最大向右移动: {right_limit}步")
    #
    #     # 3. 边界检查 - 针对第一个缓存点（即使不在位置0）
    #     if cache_point_idx == 0:
    #         # 第一个缓存点不能向左移动出界
    #         left_limit = min(left_limit, current_pos)
    #         print(f"  第一个缓存点，限制向左移动不超过: {left_limit}")
    #
    #     # 确保非负
    #     left_limit = max(0, left_limit)
    #     right_limit = max(0, right_limit)
    #
    #     print(f"  最终结果: left_limit={left_limit}, right_limit={right_limit}")
    #
    #     return left_limit, right_limit
    #
    # def _get_new_position(self, cache_point_idx, current_pos):
    #     """获取新的缓存点位置
    #
    #     Args:
    #         cache_point_idx: 缓存点在cache_index列表中的索引
    #         current_pos: 当前缓存点的位置
    #
    #     Returns:
    #         new_pos: 新的位置，如果无法移动则返回None
    #     """
    #     # 规则1：第一个缓存点（位置0）不能移动
    #     # 修正：只检查位置0
    #     if current_pos == 0:
    #         print("  ❌ 缓存点在位置0不能移动")
    #         return None
    #
    #     # 获取可移动的范围
    #     left_limit, right_limit = self._get_move_range(cache_point_idx)
    #
    #     if left_limit == 0 and right_limit == 0:
    #         print(f"  ❌ 缓存点[{cache_point_idx}] 无法移动")
    #         return None  # 无法移动
    #
    #     # 随机选择移动方向和步数
    #     if left_limit > 0 and right_limit > 0:
    #         # 两边都可移动
    #         if random.random() < 0.5:
    #             # 向左移动
    #             move_steps = random.randint(1, left_limit)
    #             new_pos = current_pos - move_steps
    #             direction = "左"
    #         else:
    #             # 向右移动
    #             move_steps = random.randint(1, right_limit)
    #             new_pos = current_pos + move_steps
    #             direction = "右"
    #     elif left_limit > 0:
    #         # 只能向左移动
    #         move_steps = random.randint(1, left_limit)
    #         new_pos = current_pos - move_steps
    #         direction = "左"
    #     else:
    #         # 只能向右移动
    #         move_steps = random.randint(1, right_limit)
    #         new_pos = current_pos + move_steps
    #         direction = "右"
    #
    #     # 规则3：最后一个缓存点不能移动到99
    #     if cache_point_idx == len(self.cache_index) - 1:
    #         # 修正：应该是 >= self.max_length - 1
    #         # 如果max_length=100，那么不能移动到99
    #         if new_pos >= self.max_length - 1:
    #             print(f"  ❌ 最后一个缓存点不能移动到位置{self.max_length - 1}或之后")
    #             return None
    #
    #     print(f"  ✅ 缓存点[{cache_point_idx}] 向{direction}移动{move_steps}步: {current_pos} → {new_pos}")
    #
    #     return new_pos

    def _get_move_range(self, cache_point_idx, mode="normal"):
        """计算缓存点可以向左和向右移动的步数"""
        # 这里 cache_index 存储的是 (位置, 深度)
        current_pos, current_depth = self.cache_index[cache_point_idx]  # current_depth 就是深度值！


        # 规则1：第一个缓存点（位置0）不能移动
        if current_pos == 0:
            print("  缓存点在位置0，不能移动")
            return 0, 0

        # 1. 向左移动的限制
        left_limit = 0
        left_depth = None
        if cache_point_idx > 0:
            # 左边有缓存点
            left_pos, left_depth = self.cache_index[cache_point_idx - 1]  # 直接获取深度值

            # 可以向左移动的最大步数 = 当前时间步 - 左边时间步 - 1
            left_limit = current_pos - left_pos - 1

            # print(f"  左边缓存点: ({left_pos}, {left_depth})")
            # print(f"  需要保持至少1个0值间隔: {current_pos} - {left_pos} - 1 = {left_limit}")

        # 2. 向右移动的限制
        right_limit = 0
        right_depth = None
        if cache_point_idx < len(self.cache_index) - 1:
            # 右边有缓存点
            right_pos, right_depth = self.cache_index[cache_point_idx + 1]  # 直接获取深度值

            # 可以向右移动的最大步数 = 右边时间步 - 当前时间步 - 1
            right_limit = right_pos - current_pos - 1

            # print(f"  右边缓存点: ({right_pos}, {right_depth})")
            # print(f"  需要保持至少1个0值间隔: {right_pos} - {current_pos} - 1 = {right_limit}")
        else:
            # 这是最后一个缓存点
            max_right = self.max_length - current_pos - 2
            right_limit = max(0, max_right)

            # print(f"  最后一个缓存点，限制向右移动不超过位置{self.max_length - 2}")
            # print(f"  最大向右移动: {right_limit}步")

        # 3. 边界检查
        if cache_point_idx == 0:
            left_limit = min(left_limit, current_pos)
            # print(f"  第一个缓存点，限制向左移动不超过: {left_limit}")

        # 4. 根据模式调整移动限制
        if mode == "heavier":
            # heavier模式：depth小=重，depth大=轻
            if left_depth is not None:
                if left_depth < current_depth:
                    # 左边depth更小（更重），只能向右移动
                    # print(f"  heavier模式: 左边depth={left_depth} < 当前depth={current_depth} (更重)，只能向右移动")
                    left_limit = 0
                elif left_depth > current_depth:
                    # 左边depth更大（更轻），只能向左移动
                    # print(f"  heavier模式: 左边depth={left_depth} > 当前depth={current_depth} (更轻)，只能向左移动")
                    right_limit = 0
                else:
                    pass
                    # print(f"  heavier模式: 左边depth={left_depth} = 当前depth={current_depth}，两边都可以移动")

        elif mode == "lighter":
            # lighter模式：depth大=轻，depth小=重
            if left_depth is not None:
                if left_depth > current_depth:
                    # 左边depth更大（更轻），只能向右移动
                    # print(f"  lighter模式: 左边depth={left_depth} > 当前depth={current_depth} (更轻)，只能向右移动")
                    left_limit = 0
                elif left_depth < current_depth:
                    # 左边depth更小（更重），只能向左移动
                    # print(f"  lighter模式: 左边depth={left_depth} < 当前depth={current_depth} (更重)，只能向左移动")
                    right_limit = 0
                else:
                    # print(f"  lighter模式: 左边depth={left_depth} = 当前depth={current_depth}，两边都可以移动")
                        pass
        # 确保非负
        left_limit = max(0, left_limit)
        right_limit = max(0, right_limit)

        # print(f"  最终结果: left_limit={left_limit}, right_limit={right_limit}")

        return left_limit, right_limit

    def _get_new_position(self, cache_point_idx, current_pos, mode="normal"):
        """获取新的缓存点位置

        Args:
            cache_point_idx: 缓存点在cache_index列表中的索引
            current_pos: 当前缓存点的位置
            mode: 移动模式 - "normal", "heavier", "lighter"

        Returns:
            new_pos: 新的位置，如果无法移动则返回None
        """
        # 规则1：第一个缓存点（位置0）不能移动
        if current_pos == 0:
            # print(f"  ❌ 缓存点[{cache_point_idx}]在位置0不能移动 (模式={mode})")
            return None

        # 获取可移动的范围
        left_limit, right_limit = self._get_move_range(cache_point_idx, mode)

        if left_limit == 0 and right_limit == 0:
            # print(f"  ❌ 缓存点[{cache_point_idx}] 无法移动 (模式={mode})")
            return None  # 无法移动

        # 根据可移动的方向选择
        if left_limit > 0 and right_limit > 0:
            # 两边都可移动
            if random.random() < 0.5:
                # 向左移动
                move_steps = random.randint(1, left_limit)
                new_pos = current_pos - move_steps
                direction = "左"
            else:
                # 向右移动
                move_steps = random.randint(1, right_limit)
                new_pos = current_pos + move_steps
                direction = "右"
        elif left_limit > 0:
            # 只能向左移动
            move_steps = random.randint(1, left_limit)
            new_pos = current_pos - move_steps
            direction = "左"
        else:
            # 只能向右移动
            move_steps = random.randint(1, right_limit)
            new_pos = current_pos + move_steps
            direction = "右"

        # 规则3：最后一个缓存点不能移动到99
        if cache_point_idx == len(self.cache_index) - 1:
            if new_pos >= self.max_length - 1:
                # print(f"  ❌ 最后一个缓存点不能移动到位置{self.max_length - 1}或之后")
                return None

        # print(f"  ✅ 缓存点[{cache_point_idx}] 向{direction}移动{move_steps}步: {current_pos} → {new_pos} (模式={mode})")
        return new_pos

    def _get_new_depth(self, current_depth, mode="normal"):
        """根据模式获取新的深度值"""
        if mode == "normal":
            # 正常模式：随机选择1-4中不等于当前深度的值
            candidate_depths = [d for d in range(1, 5) if d != current_depth]
        elif mode == "heavier":
            # 加重模式：只能选择比当前小的深度值（深度越小越重）
            candidate_depths = [d for d in range(1, current_depth)] if current_depth > 1 else []
        elif mode == "lighter":
            # 减轻模式：只能选择比当前大的深度值（深度越大越轻）
            candidate_depths = [d for d in range(current_depth + 1, 5)] if current_depth < 4 else []
        else:
            candidate_depths = []

        if candidate_depths:
            new_depth = random.choice(candidate_depths)
            # print(f"  Depth change: {current_depth} -> {new_depth} (mode={mode}, candidates={candidate_depths})")
            return new_depth
        else:
            # print(f"  No valid depth change for {current_depth} in {mode} mode")
            return current_depth  # 没有可选的深度，返回原值

        # 如果没有候选深度，返回原深度
        if not candidate_depths:
                return current_depth

        return random.choice(candidate_depths)

    def _update_ms_from_cache_index(self):
        """根据cache_index更新ms序列

        规则：
        1. 缓存点位置为0
        2. 从每个缓存点开始，用该缓存点的深度值填充后续位置
        3. 直到遇到下一个缓存点为止（不包含下一个缓存点）
        """
        if self.cache_index is None:
            return

        # 创建全0序列
        ms = np.zeros(self.max_length, dtype=int)

        # 对缓存点按位置排序（确保有序）
        sorted_cache_points = sorted(self.cache_index, key=lambda x: x[0])

        # 遍历每个缓存点
        for i, (cache_pos, depth) in enumerate(sorted_cache_points):
            # 1. 设置缓存点本身为0（已经是0）

            # 2. 确定当前缓存点覆盖的结束位置
            if i < len(sorted_cache_points) - 1:
                # 不是最后一个缓存点，结束于下一个缓存点之前
                next_cache_pos = sorted_cache_points[i + 1][0]
                end_pos = next_cache_pos
            else:
                # 最后一个缓存点，覆盖到序列末尾
                end_pos = self.max_length

            # 3. 从缓存点之后的位置开始填充深度值
            # 注意：缓存点本身保持为0，从cache_pos+1开始填充
            start_fill_pos = cache_pos + 1

            # 确保不超过序列边界
            if start_fill_pos < end_pos:
                for pos in range(start_fill_pos, end_pos):
                    ms[pos] = depth

        # 设置缓存点位置为0（已经是0，但再次确认）
        for cache_pos, _ in sorted_cache_points:
            if cache_pos < self.max_length:
                ms[cache_pos] = 0

        self.set_ms(ms)

        # 验证：应该有正好 len(cache_index) 个0值
        zero_count = np.sum(self.ms == 0)
        expected_zero_count = len(self.cache_index)

        if zero_count != expected_zero_count:
            print(f"⚠️ Warning: ms序列有{zero_count}个0，但应该有{expected_zero_count}个")
            print(f"Cache points: {sorted_cache_points}")
            print(f"MS sequence: {self.ms}")

    def validate_ms_sequence(self):
        """验证ms序列的正确性"""
        if self.cache_index is None or self.ms is None:
            return False

        # 1. 检查0值数量等于缓存点数量
        zero_count = np.sum(self.ms == 0)
        expected_zero_count = len(self.cache_index)

        if zero_count != expected_zero_count:
            print(f"❌ 错误: 0值数量不匹配。有{zero_count}个0，应有{expected_zero_count}个")
            return False

        # 2. 检查每个缓存点位置确实是0
        for cache_pos, _ in self.cache_index:
            if cache_pos >= len(self.ms):
                print(f"❌ 错误: 缓存点位置{cache_pos}超出序列范围")
                return False
            if self.ms[cache_pos] != 0:
                print(f"❌ 错误: 缓存点位置{cache_pos}不是0，而是{self.ms[cache_pos]}")
                return False

        # 3. 检查深度值填充正确
        sorted_cache_points = sorted(self.cache_index, key=lambda x: x[0])

        for i, (cache_pos, depth) in enumerate(sorted_cache_points):
            # 确定当前段的范围
            if i < len(sorted_cache_points) - 1:
                next_cache_pos = sorted_cache_points[i + 1][0]
                end_pos = next_cache_pos
            else:
                end_pos = len(self.ms)

            # 检查从cache_pos+1到end_pos-1是否都是depth
            for pos in range(cache_pos + 1, end_pos):
                if self.ms[pos] != depth:
                    print(f"❌ 错误: 位置{pos}应该是{depth}，但却是{self.ms[pos]}")
                    print(f"  缓存点{cache_pos}深度{depth}应覆盖到{end_pos - 1}")
                    return False

        print("✅ ms序列验证通过")
        return True

    def check(self):
        return True
    
class controller:
    def __init__(self, config, predictor, logger):
        # search configurations
        self.config = config
        self.sampler_type = config["predictor"]["sampler_type"]
        self.smaller_score = 1 if config["search"]["smaller_score"] else -1
        self.max_init_time = config["search"]["max_init_time"]
        self.max_num_next_generation = config["search"]["max_num_next_generation"]
        self.max_mutate_time_one_iter = config["search"]["max_mutate_time_one_iter"]
        self.init_tolerance = config["search"]["init_tolerance"]
        self.max_candidate_parents = config["search"]["max_candidate_parents"]
        self.max_population_size = config["search"]["max_population_size"]
        self.epoch = config["search"]["epoch"]
        self.log_every = config["search"]["log_every"]
        
        # predictor
        self.predictor = predictor
        
        # logger
        self.logger = logger
    
    def get_initial_ms(self, budget, init_ms=None):
        if self.sampler_type=="dpm-solver":
            ms = dpmsolver_model_schedule(self.config)
        elif self.sampler_type=="ddim":
            ms = ddim_model_schedule(self.config)
        else:
            raise NotImplementedError(f"Sampler type {self.sampler_type} is not supported!")
        
        if init_ms is not None:
            ms.set_ms(init_ms)
        else:
            ms.random_init_ms(tolerance=self.init_tolerance, max_init_time=self.max_init_time, budget=budget)
            
        return ms

    # def step(self, budget):
    #     '''
    #     Generate the next generation of model schedules.
    #     '''
    #     next_generation = []
    #     mutate_time = 0
    #     while(1):
    #         # get candidates
    #         indices = random.sample(range(0, len(self.population)), self.max_candidate_parents) if len(self.population)>self.max_candidate_parents else range(0, len(self.population))
    #
    #         # get parent
    #         best_index = min(indices, key = lambda x: self.smaller_score * self.population[x].get_perf(self.predictor))
    #         parent = self.population[best_index]
    #
    #         # mutate
    #         new_ms = parent.mutate()
    #
    #         if new_ms.get_cost()<=budget: # abandon model schedules with latencies exceeding the budget
    #             next_generation.append(new_ms)
    #
    #         mutate_time += 1
    #
    #         if len(next_generation)==self.max_num_next_generation or mutate_time==self.max_mutate_time_one_iter:
    #             break
    #
    #     return next_generation

    def step(self, budget):
        '''
        Generate the next generation of model schedules.
        '''
        next_generation = []
        mutate_time = 0
        while (1):
            # get candidates
            # 锦标赛选择：随机选k个，从中选最好的
            tournament_size = 5 # 锦标赛大小

            # get parent
            # 随机选择tournament_size个候选
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            # 从锦标赛中选择最好的
            best_index = min(tournament_indices,key=lambda x: self.smaller_score * self.population[x].get_perf(self.predictor))
            parent = self.population[best_index]

            # mutate
            new_ms = parent.mutate()

            if new_ms.get_cost() <= budget:  # abandon model schedules with latencies exceeding the budget
                next_generation.append(new_ms)

            mutate_time += 1

            if len(next_generation) == self.max_num_next_generation or mutate_time == self.max_mutate_time_one_iter:
                break

        return next_generation

    # def save_population(self, save_path):
    #     clean_ms_population = []
    #     for i in range(len(self.population)):
    #         clean_ms_population.append(self.population[i].ms.tolist())
    #     torch.save(clean_ms_population, save_path)

    def save_population(self, save_path):
        clean_ms_population = []
        clean_ms_population.append(self.population[0].ms.tolist())
        torch.save(clean_ms_population, save_path)
            
    # def search(self, budget, save_path,seed):
    #     initial_ms = self.get_initial_ms(budget)
    #     # 这里的population是schedule的集合
    #     self.population = [initial_ms]
    #     # 生成多样化的初始种群
    #     initial_population = []
    #     for _ in range(self.max_population_size - 20):
    #         ms = self.get_initial_ms(budget)
    #         initial_population.append(ms)
    #
    #     self.population = initial_population
    #     print(f"初始种群大小: {len(self.population)}")
    #
    #     for i in range(self.epoch):
    #         # generate the next generation
    #         next_generation = self.step(budget)
    #         self.population += next_generation
    #
    #         # eliminate the individuals with poor performance, keeping the size of population smaller than a particular value
    #         self.population.sort(key=lambda x: self.smaller_score * x.get_perf(self.predictor))
    #         if len(self.population)>self.max_population_size:
    #             self.population = self.population[:self.max_population_size]
    #
    #         # print info
    #         best_score = self.population[0].get_perf(self.predictor)
    #         if i % self.log_every==1:
    #             self.logger.info(f"Epoch {i} | best predict score {best_score}")
    #
    #     print(self.population[0].ms)
    #     print(self.population[0].get_perf(self.predictor))
    #     self.save_population(os.path.join(save_path, "final_population.pth"))
    # 
    def search(self, budget, save_path,seed):
        # 生成多样化的初始种群
        initial_population = []
        for _ in range(self.max_population_size ):
            ms = self.get_initial_ms(budget) # 记得调整这里面的参数，做消融实验
            initial_population.append(ms)

        self.population = initial_population
        print(f"初始种群大小: {len(self.population)}")

        for i in range(self.epoch):
            # generate the next generation
            next_generation = self.step(budget) # 记得调整这里面的参数，做消融实验
            self.population += next_generation

            # eliminate the individuals with poor performance, keeping the size of population smaller than a particular value
            self.population.sort(key=lambda x: self.smaller_score * x.get_perf(self.predictor))
            if len(self.population)>self.max_population_size:
                self.population = self.population[:self.max_population_size]

            # print info
            best_score = self.population[0].get_perf(self.predictor)
            worst_score = self.population[-1].get_perf(self.predictor)
            if i % self.log_every==1:
                self.logger.info(f"Epoch {i} | best predict score {best_score} | worst predict score {worst_score}")

        print(self.population[0].ms)
        print(self.population[0].get_perf(self.predictor))
        # 构建包含budget和seed的文件名
        # 如果seed不存在，使用默认值或当前时间戳
        # population_file ="final_population.pth"
        population_file = f"../cifar_Ablation_Study_population/final_population_budget{budget}_seed{seed}.pth"
        self.save_population(os.path.join(save_path, population_file))

'''
ours-消融实验
时间步搜寻，分别固定四个缓存位置，时间步分布baseline选取 uniform ，根据四个不同预算搜寻时间步，仅仅只能选择移动变异（不改变开销）

缓存位置搜寻，固定时间步分布为 uniform ，缓存位置baseline选取  四个缓存位置，根据四个不同预算搜寻时间步，仅仅只能选择深度变异（要改变开销，根据cost筛选）
3 or 4?
'''