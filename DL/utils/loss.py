import torch

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def focal_loss(pred, target, gamma=2):
    pred = pred.view(-1)
    target = target.view(-1)
    return -torch.mean((1 - pred) ** gamma * target * torch.log(pred + 1e-8) +
                       pred ** gamma * (1 - target) * torch.log(1 - pred + 1e-8))

def tv_loss(pred):
    return (torch.mean(torch.abs(pred[:, :, :-1] - pred[:, :, 1:])) +
            torch.mean(torch.abs(pred[:, :-1, :] - pred[:, 1:, :])))

def total_loss(pred, target):
    return 0.7 * dice_loss(pred, target) + 0.3 * focal_loss(pred, target) + 0.1 * tv_loss(pred)
