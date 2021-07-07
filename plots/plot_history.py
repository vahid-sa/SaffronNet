from os import path
from matplotlib import pyplot as plt
import json


def plot_loss_mAP(json_file_path: str):
    fileIO = open(json_file_path, "r")
    rows = fileIO.readlines()
    fileIO.close()
    # epochs = list()
    class_losses = list()
    xyreg_losses = list()
    alphareg_losses = list()
    mAP = list()
    num_annots = list()
    for row in rows:
        info = json.loads(row)
        # epochs.append(int(info["epoch"]))
        class_losses.append(float(info["c-loss"]))
        xyreg_losses.append(float(info["rxy-loss"]))
        alphareg_losses.append(float(info["ra-loss"]))
        d = info["mAp"]
        d = d[d.index("(") + 1:d.index(")")].split(", ")
        d = float(d[0]), int(float(d[1]))
        mAP.append(d[0])
        num_annots.append(d[1])
    epochs = list(range(len(rows)))
    fig, ax = plt.subplots(2, 2, figsize=(12, 4))
    ax[0, 0].plot(epochs, class_losses, color="red")
    ax[0, 0].set_ylabel("classification loss")
    ax[0, 0].set_xlabel("epoch")

    ax[0, 1].plot(epochs, xyreg_losses, color="red")
    ax[0, 1].set_ylabel("XY regression loss")
    ax[0, 1].set_xlabel("epoch")

    ax[1, 0].plot(epochs, alphareg_losses, color="red")
    ax[1, 0].set_ylabel("angle regression loss")
    ax[1, 0].set_xlabel("epoch")

    ax[1, 1].plot(epochs, mAP, color="blue")
    ax[1, 1].set_ylabel("mAP")
    ax[1, 1].set_xlabel("epoch")
    # ax[1, 1].set_ylim([0.0 - 0.05, 1.0 + 0.05])

    fig.suptitle("Number of Annotated Saffrons = {0}\nsupervised".format(num_annots[0]))
    plt.show()


PATH = './history.json'
plot_loss_mAP(json_file_path=PATH)
