import os
import json
import numpy as np
from matplotlib import pyplot as plt
from os import path as osp

def lod2dol(lod: list) -> dict:  # list of dicts to dict of lists
    keys = lod[0].keys()
    dol = {key: [] for key in keys}
    for d in lod:
        for key in keys:
            dol[key].append(d[key])
    return dol


def get_mAP_per_cycle(cycles: list, mAPs: list, type: str) -> list:
    mAPs = np.array(mAPs)
    unique_cycles = np.sort(np.unique(cycles))
    mAPs_per_cycle = []
    for cycle in unique_cycles:
        indices = np.where(cycles == cycle)
        mAPs_per_cycle.append(mAPs[indices])
    if type == "max":
        mAP_per_cycle = [max(mAP) for mAP in mAPs_per_cycle]
    elif type == "avg":
        mAP_per_cycle = [np.mean(mAP) for mAP in mAPs_per_cycle]
    elif type == "last":
        mAP_per_cycle = [mAP[-1] for mAP in mAPs_per_cycle]
    else:
        raise AssertionError("incorrect option type")
    return mAP_per_cycle



def get_loss_per_cycle(cycles: list, losses: list, type: str) -> list:
    losses = np.array(losses)
    unique_cycles = np.sort(np.unique(cycles))
    losses_per_cycle = []
    for cycle in unique_cycles:
        indices = np.where(cycles == cycle)
        losses_per_cycle.append(losses[indices])
    if type == "max":
        loss_per_cycle = [max(loss) for loss in losses_per_cycle]
    elif type == "avg":
        loss_per_cycle = [np.mean(loss) for loss in losses_per_cycle]
    elif type == "last":
        loss_per_cycle = [loss[-1] for loss in losses_per_cycle]
    else:
        raise AssertionError("incorrect option type")
    return loss_per_cycle


path = osp.expanduser("~/Documents/thesis/metrics/image_level_least_avg.json")
f = open(path, "r")
f_str = f.read()
f.close()
metrics = json.loads(f_str)
active_metrics = lod2dol(metrics['active'])  # num_cycle, num_labels, num_images
train_metrics = lod2dol(metrics['train'])  # cycle, epoch, mAP, loss, lr
epochs = train_metrics['epoch']
cycles = train_metrics['cycle']
cycles = [c - 1 for c in cycles] if min(cycles) > 0 else cycles
poses = [cycles.index(item) for item in np.unique(cycles)]
mAPs = get_mAP_per_cycle(cycles=train_metrics['cycle'], mAPs=train_metrics['mAP'], type='avg')
losses = get_loss_per_cycle(cycles=train_metrics['cycle'], losses=train_metrics['loss'], type="last")
x = np.unique(train_metrics['cycle']).tolist()
plt.plot(x, mAPs)
# for pos in poses:
#     plt.axvline(pos)
plt.show()
# plt.plot(range(len(train_metrics["mAP"])), train_metrics['mAP'])
# plt.show()
