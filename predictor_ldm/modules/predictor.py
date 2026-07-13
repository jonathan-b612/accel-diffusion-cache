import torch
import torch.nn as nn
import importlib

from modules.model_embedder import *
from modules.ms_encoder import *
from modules.regression_head import *
from modules.timestep_encoder import *
from modules.predictor_utils import *
from modules.noise_schedules import *

def get_predictor(config, noise_schedule=None):
    sampler_type = config["sampler_type"]
    if sampler_type=="dpm-solver":
        predictor = ms_predictor_dpm_solver(config, noise_schedule)
    elif sampler_type=="ddim":
        predictor = ms_predictor_ddim(config)
    else:
        raise NotImplementedError(f"Sampler type {sampler_type} is not supported")
    return predictor

def load_ckpt(args, device, predictor, optimizer):
    state_dict = torch.load(args.resume, map_location=device)
    if isinstance(state_dict, dict):
        predictor.load_state_dict(state_dict["predictor"])
        optimizer.load_state_dict(state_dict["optimizer"])
        epoch = state_dict["epoch"]
    elif isinstance(state_dict, list):
        predictor.load_state_dict(state_dict[0])
        optimizer.load_state_dict(state_dict[1])
        epoch = state_dict[2]
    return epoch

class ms_predictor(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        # model embedder
        self.model_embedder = model_embedder(config["model_embedder"])

        # timestep encoder
        self.timestep_encoder = timestep_encoder(config["timestep_encoder"])

        # model sequence encoder (LSTM)
        # Since the dimensions of the model embedder of the predictor of two samplers are not the same, it is left to be defined in the corresponding subclasses

        # score regression head
        input_size = config["ms_encoder"]["hidden_size"]
        self.regression_head = regression_head(config["regression_head"], input_size)

        # loss
        self.loss_type = config["loss"]["loss_type"]
        self.compare_threshold = config["loss"]["ranking"]["compare_threshold"]
        self.max_compare_ratio = config["loss"]["ranking"]["max_compare_ratio"]

    def forward(self):
        pass

    def cal_loss(self, data, gt_score):
        if self.loss_type=="ranking":
            loss = self.update_compare(data, gt_score)
        elif self.loss_type=="mse":
            loss = self.update_predict(data, gt_score)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} is not supported currently!")
        return loss

    def update_compare(self, data, gt_score):
        data_1, data_2, better_lst = compare_data(data, gt_score, self.compare_threshold, self.max_compare_ratio)
        s_1 = self.forward(data_1)
        s_2 = self.forward(data_2)
        better_pm = 2 * s_1.new(np.array(better_lst, dtype=np.float32)) - 1
        zero_ = s_1.new([0.])
        margin = self.config["loss"]["ranking"]["compare_margin"]
        margin = s_1.new([margin])
        pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        return pair_loss

    def update_predict(self, data, gt_score):
        pred_score = self.forward(data)
        return (pred_score - gt_score).square().mean()

class ms_predictor_ddim(ms_predictor):
    def __init__(self, config):
        super().__init__(config)
        self.max_timesteps = config["max_timesteps"]

        # model sequence encoder (LSTM)
        input_size = config["timestep_encoder"]["output_temb_dim"] + config["model_embedder"]["embedding_dim"]
        self.ms_encoder = ms_encoder(config["ms_encoder"], input_size)

    def forward(self, data):
        '''
        Args:
            ms: ddim model schedule. shape: [bs, L] (see eq.6 in our original paper)
        '''
        # get input data and set device
        ms = data["ms"].to(list(self.parameters())[0].device)

        # get model embedding sequence
        model_emb = self.model_embedder(ms)

        # get timesteps
        # timesteps = get_timesteps_for_ddim(ms.shape[1], self.max_timesteps).to(ms.device)
        # timesteps = get_timesteps_for_ddim_quad(ms.shape[1], self.max_timesteps).to(ms.device)
        timesteps = get_timesteps_for_ddim_uniform(ms.shape[1], self.max_timesteps).to(ms.device)
        # get timestep embeddings
        timestep_emb = self.timestep_encoder(timesteps)
        if len(timestep_emb.shape)==2:
            timestep_emb = timestep_emb.unsqueeze(0).repeat(ms.size(0), 1, 1)

        # concat model embeddings and timestep embedding
        whole_emb = torch.cat([model_emb, timestep_emb], dim=2)

        # get overall model schedule embedding
        ms_emb = self.ms_encoder(whole_emb)

        # regress the final score
        score = self.regression_head(ms_emb)

        return score
