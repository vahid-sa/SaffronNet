import os
import json
import numpy as np
from os import path as osp
from matplotlib import pyplot as plt

path = "/run/user/1000/gvfs/smb-share:server=172.17.3.145,share=saeedi97/tmp/saffron_imgs/metrics.json"

f = open(path, "r")
string = f.read()
f.close()
metrics = json.loads(string)
maps = metrics['mAP']
losses = metrics["loss"]
cycles = np.unique(metrics['cycle'])
epochs = np.unique(metrics['epoch'])
cycles = cycles - 1 if min(cycles) > 0 else cycles
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(np.arange(len(maps)), maps, color='green')
axs[0].title.set_text("mAP")
axs[1].plot(np.arange(len(losses)), losses, color='red')
axs[1].title.set_text("loss")
for cyl in cycles:
    axs[0].axvline(cyl * len(epochs))
    axs[1].axvline(cyl * len(epochs))
fig.suptitle("metrics")
fig.show()
plt.show()
