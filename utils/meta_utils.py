import os
import torch
from os import path as osp


def save_models(model_path: str,
                state_dict_path: str,
                model,
                optimizer=None,
                scheduler=None,
                loss=None,
                mAP=None,
                epoch=None):
    os.makedirs(osp.dirname(model_path), exist_ok=True)
    os.makedirs(osp.dirname(state_dict_path), exist_ok=True)
    state_dict = {"model_state_dict": model.state_dict(), }
    if optimizer is not None:
        state_dict['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    if loss is not None:
        state_dict["loss"] = loss
    if mAP is not None:
        state_dict["mAP"] = mAP
    if epoch is not None:
        state_dict["epoch"] = epoch
    with open(model_path, "wb") as fileIO:
        torch.save(obj=model, f=fileIO)
    with open(state_dict_path, "wb") as fileIO:
        torch.save(obj=state_dict, f=fileIO)
