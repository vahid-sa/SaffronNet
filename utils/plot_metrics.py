import os
import json
import numpy as np
from os import path as osp
from matplotlib import pyplot as plt
from copy import deepcopy

path = "/home/vahid/metrics.json"

f = open(path, "r")
string = f.read()
f.close()
metrics = json.loads(string)
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
nnl = deepcopy(counts["num_new_labels"])
nnl.pop(-1)
nnl.insert(0, 0)
counts["num_new_labels"] = [c - n for c, n in zip(counts["num_new_labels"], nnl)]
fig, axs = plt.subplots(3, 1, figsize=(12, 4))
axs[0].bar(x=counts['num_cycle'], height=counts['num_new_labels'])
axs[1].bar(x=counts['num_cycle'], height=counts['num_higher_half_queries'])
axs[1].bar(x=counts['num_cycle'], height=counts['num_lower_half_queries'])
fig.suptitle("counts")
fig.show()
plt.show()