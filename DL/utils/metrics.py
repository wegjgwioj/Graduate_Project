import numpy as np
from sklearn.metrics import jaccard_score

def calculate_miou(pred, target):
    pred_binary = (pred > 0.5).astype(np.uint8)
    target_binary = target.astype(np.uint8)
    return jaccard_score(target_binary.flatten(), pred_binary.flatten())

def calculate_fps(inference_time, num_frames):
    return num_frames / inference_time