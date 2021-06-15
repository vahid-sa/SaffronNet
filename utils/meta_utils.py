import torch


def save_models(model_path: str,
                state_dict_path: str,
                model,
                optimizer=None,
                scheduler=None,
                loss=None,
                mAP=None,
                epoch=None):
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

    torch.save(model, model_path)
    torch.save(state_dict, state_dict_path)
