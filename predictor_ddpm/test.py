import os

import torch
from scipy.stats import kendalltau

def cal_kendall_tau(set1, set2):
    corr = kendalltau(set1, set2).correlation
    return corr

def valid_one_epoch(predictor, valid_loader, logger, set="valid set"):
    predictor.eval()
    with torch.no_grad():
        gt_scores = []
        predict_scores = []
        for ms, gt_score in valid_loader:
            predict_score = predictor(ms)
            for i in range(len(predict_score)):
                gt_scores.append(gt_score[i].item())
                predict_scores.append(predict_score[i].item())
    kd = cal_kendall_tau(gt_scores, predict_scores)
    # 只打印前20条数据
    logger.info(f"First 20 predict scores: {predict_scores[:20]}")
    logger.info(f"First 20 gt scores: {gt_scores[:20]}")
    logger.info(f"The KD of predict scores and gt scores on {set} is {kd}")
    return kd
    
def test(predictor, valid_loader, logger,model_path = None):
    # 如果没有提供模型路径，尝试默认路径
    if model_path is None:
        # 你需要从配置或其他地方获取默认路径
        model_path = "randomddpm/predictor/exps/debug/final.pth"  # 这里需要调整

    # 加载训练好的模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
        predictor.load_state_dict(checkpoint["predictor"])
        logger.info(f"Loaded model from {model_path}")

    valid_one_epoch(predictor, valid_loader, logger)