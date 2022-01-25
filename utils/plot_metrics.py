import os
import json
import numpy as np
from os import path as osp
from matplotlib import pyplot as plt
from copy import deepcopy

path = "/home/vahid/Documents/thesis/metrics/box_level_least.json"

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

f = open(path, "r")
string = f.read()
f.close()
metrics = json.loads(string)

mAPs = metrics['mAP']
cycles = metrics['cycle']
counts = {key: [dic[key] for dic in metrics['annotations']] for key in metrics['annotations'][0]}
num_labels = counts['num_labels']
map_per_cycle = get_mAP_per_cycle(cycles=cycles, mAPs=mAPs, type='avg')
plt.plot(num_labels, map_per_cycle)
plt.show()


maps = metrics['mAP']
losses = metrics["loss"]
lrs = metrics['lr']
# map_train = metrics['mAP_train']
cycles = np.unique(metrics['cycle'])
epochs = np.unique(metrics['epoch'])
cycles = cycles - 1 if min(cycles) > 0 else cycles
fig, axs = plt.subplots(3, 1, figsize=(12, 4))
axs[0].plot(np.arange(len(maps)), maps, color='green')
axs[0].title.set_text("mAP")
axs[1].plot(np.arange(len(losses)), losses, color='red')
axs[1].title.set_text("loss")
axs[2].plot(np.arange(len(lrs)), lrs, color='blue')
axs[2].title.set_text("learning rate")
# axs[3].plot(np.arange(len(map_train)), map_train)
# axs[3].title.set_text("mAP train")
for cyl in cycles:
    axs[0].axvline(cyl * len(epochs))
    axs[1].axvline(cyl * len(epochs))
    axs[2].axvline(cyl * len(epochs))
    # axs[3].axvline(cyl * len(epochs))

fig.suptitle("metrics")
fig.show()
plt.show()


counts = {key: [dic[key] for dic in metrics['annotations']] for key in metrics['annotations'][0]}
print(counts.keys())
nnl = deepcopy(counts["num_labels"])
nnl.pop(-1)
nnl.insert(0, 0)
counts["num_labels"] = [c - n for c, n in zip(counts["num_labels"], nnl)]
fig, axs = plt.subplots(4, 1, figsize=(15, 4))
axs[0].bar(x=counts['num_cycle'], height=counts['num_labels'])
axs[1].bar(x=counts['num_cycle'], height=counts['num_higher_half_queries'])
axs[1].bar(x=counts['num_cycle'], height=counts['num_lower_half_queries'])
fig.suptitle("counts")
fig.show()
plt.show()
