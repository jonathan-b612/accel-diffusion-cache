import torch
import random
import numpy

from search_utils import *

def search(args, config, predictor, logger,model_path = None):
    assert args.budget is not None, "need to specify a budget to constrain the search"

    # 如果没有提供模型路径，尝试默认路径
    if model_path is None:
        # 你需要从配置或其他地方获取默认路径
        model_path = "randomldm/predictor/exps/debug/final.pth"  # 这里需要调整

    # 加载训练好的模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
        predictor.load_state_dict(checkpoint["predictor"])
        logger.info(f"Loaded model from {model_path}")
    predictor.eval()
    
    evo_controller = controller(config, predictor, logger)
    evo_controller.search(args.budget, args.exp,args.seed)
    
    print("Search complete")