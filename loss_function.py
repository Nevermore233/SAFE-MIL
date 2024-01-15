import numpy as np


######################## Loss Function ########################
def HL_loss(out, target):
    out_max = np.max(out)
    if out_max < 0:
        out_max = 0
    if out_max > 1:
        out_max = 1
    loss = (out_max - target) * (out_max - target) / out_max
    return loss


def huber_loss(out, target, delta):
    out_max = np.max(out)
    error = out_max - target
    squared_loss = 0.5 * error**2
    absolute_loss = delta * (np.abs(error) - 0.5 * delta)
    loss = np.where(np.abs(error) <= delta, squared_loss, absolute_loss)
    return loss


def mse_loss(out, target):
    out_max = np.max(out)
    loss = (out_max - target) * (out_max - target) * 0.5
    return loss


def mae_loss(out, target):
    out_max = np.max(out)
    loss = abs(out_max - target)
    return loss


def log_cosh_loss(out, target):
    out_max = np.max(out)
    loss = np.log(np.cosh(out_max - target))
    return loss
