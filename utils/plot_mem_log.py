import json
import numpy as np
from matplotlib import pyplot as plt

path = "/run/user/1000/gvfs/smb-share:server=172.17.3.145,share=saeedi97/Safffron/memory_status.json"
f = open(path, "r")
d = json.loads(f.read())
f.close()
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

cuda_error_indices = np.squeeze(np.argwhere(d['cuda_error']))
for i , name in enumerate(('used', 'free', 'total')):
    for index in cuda_error_indices:
        axs[i].axvline(index, color='red')
    axs[i].bar(x=np.arange(len(d[name])), height=d[name])
    axs[i].title.set_text(name)
fig.show()
plt.show()
