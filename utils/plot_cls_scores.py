import numpy as np
from matplotlib import pyplot as plt
# import proplot as plt
import json
path = '/run/user/1000/gvfs/smb-share:server=172.17.3.145,share=saeedi97/tmp/saffron_imgs/classification_scores'
f = open(path, 'r')
string = f.read()
f.close()
rows = string.replace("}", "}\n").splitlines()
rows = [json.loads(row) for row in rows]
classifications_hist = {}
for row in rows:
    cycle = row['cycle']
    epoch = row['epoch']
    scores = row['scores']
    if not (cycle in classifications_hist):
        classifications_hist[cycle] = {}
    if not (epoch in classifications_hist[cycle]):
        l = [0.0] * 10
        classifications_hist[cycle][epoch] = l
    classifications_hist[cycle][epoch] = [a + b for (a, b) in zip(classifications_hist[cycle][epoch], scores)]

edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
width = np.diff(edges)
edges = edges[:-1]
cycles = len(classifications_hist.keys())
epochs = len(classifications_hist[1].keys())
fig, axs = plt.subplots(cycle, epochs, figsize=(18, 9))
rowlabels, collabels = [], []
start = 4
for i, cyl in enumerate(classifications_hist.keys()):
    for j, epc in enumerate(classifications_hist[cyl].keys()):
        hist = classifications_hist[cyl][epc]
        axs[i, j].bar(edges[start:], hist[start:], width=width[start:], edgecolor="black", align="edge")
        if i == 0:
            axs[i, j].title.set_text(f"Epoch {epc}")
        if j == 0:
            axs[i, j].set_ylabel(f"Cycle {cyl}")
        axs[i, j].set_xticks(edges)
fig.show()
plt.show()
