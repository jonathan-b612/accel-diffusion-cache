def forward(self, x, t, context, **kwargs):  # , prv_f=None, branch=None
    prv_f = kwargs['prv_f']
    branch = kwargs['branch']
    assert x.shape[2] == x.shape[3] == self.resolution

    # timestep embedding
    temb = get_timestep_embedding(t, self.ch)
    temb = self.temb.dense[0](temb)
    temb = nonlinearity(temb)
    temb = self.temb.dense[1](temb)

    # 这里的if分支是运用缓存的路径
    if prv_f is not None:
        print(f"=== 缓存路径验证-利用缓存 ===")
        features = None
        hs = [self.conv_in(x)]

        # 下采样阶段 - 根据select_layer和select_block终止下采样
        early_break = False
        downsample_feature_index = -1  # 初始化为-1，表示未设置
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

                # 检查是否到达下采样终止位置
                if i_level == self.select_layer and i_block == self.select_block:
                    early_break = True
                    break

            # 检查是否需要提前终止层级循环
            if early_break:
                break

            # 执行下采样（如果需要）
            if i_level != self.num_resolutions - 1:
                # 执行下采样
                downsample_output = self.down[i_level].downsample(hs[-1])
                hs.append(downsample_output)

                # 特殊处理：如果select_block=-1，表示在下采样操作后停止
                if self.select_block == -1 and i_level == self.select_layer:
                    early_break = True
                    break

        # 使用缓存特征
        h = prv_f

        # 上采样阶段
        for i_level in reversed(range(self.num_resolutions)):
            if i_level > self.restart_layer:
                continue
            for i_block in range(self.num_res_blocks + 1):
                if i_level == self.restart_layer and i_block < self.restart_block:
                    continue

                # # 缓存修正（在开始处理当前块时立即执行）
                # if kwargs['prv_f'] is not None:
                #     kwargs['prv_f'] = None
                #     if self.a_list is not None:
                #         a = self.a_list[self.time].contiguous().view(1, self.a_list[self.time].size(0), 1, 1)
                #         b = self.b_list[self.time].contiguous().view(1, self.b_list[self.time].size(0), 1, 1)
                #         h = a * h + b

                # 获取跳跃连接特征
                if hs:
                    hs_last = hs.pop()

                # 上采样块计算
                if self.config.split_shortcut:
                    split_ = h.size(1)
                    h = self.up[i_level].block[i_block](torch.cat([h, hs_last], dim=1), temb, split=split_)
                else:
                    h = self.up[i_level].block[i_block](torch.cat([h, hs_last], dim=1), temb)

                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)
    else:  # downsampling
        print(f"=== 完全路径验证-保存缓存 ===")
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        features = []
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                # if i_level < 4 and self.config.split_shortcut:
                if self.config.split_shortcut:
                    split_ = h.size(1)
                else:
                    split_ = 0
                hs_last = hs.pop()

                # if i_level == self.up_select_layer and i_block == self.up_select_block:
                #     print(f"保存缓存特征: up[{i_level}].block[{i_block}], 尺寸: {h.shape}")
                #     #features.append(h.detach())
                if self.config.split_shortcut:
                    h = self.up[i_level].block[i_block](
                        torch.cat([h, hs_last], dim=1), temb, split=split_)
                else:
                    h = self.up[i_level].block[i_block](
                        torch.cat([h, hs_last], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                if i_level == self.up_select_layer and i_block == self.up_select_block:
                    # print(f"----保存缓存特征: up[{i_level}].block[{i_block}], 尺寸: {h.shape}")
                    features.append(h.detach())

            if i_level != 0:
                h = self.up[i_level].upsample(h)

    self.time = self.time + 1
    if self.time == self.timesteps:
        self.time = 0
    # end
    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)
    return h, features