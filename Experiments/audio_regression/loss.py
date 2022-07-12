import torch


def mse(x, y):
    return (x - y).pow(2).mean()


def val_mse(gt, pred):
    val_loss = mse(gt, pred)

    return {'val_loss': val_loss}
        

def der_mse(gt_grad, pred_grad):
    weights = torch.ones(gt_grad.shape[1]).to(gt_grad.device)
    der_loss = torch.mean((weights * (gt_grad - pred_grad).pow(2)).sum(-1))

    return {'der_loss': der_loss}
