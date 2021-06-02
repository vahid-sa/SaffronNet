from matplotlib import pyplot as plt
import json


def plot_loss_mAP(json_file_path: str):
    fileIO = open(json_file_path, "r")
    rows = fileIO.readlines()
    fileIO.close()
    epochs = list()
    losses = list()
    mAP = list()
    num_annots = list()
    for row in rows:
        info = json.loads(row)
        epochs.append(int(info["epoch"]))
        losses.append(float(info["loss"]))
        d = info["mAp"]
        d = d[d.index("(") + 1:d.index(")")].split(", ")
        d = float(d[0]), int(float(d[1]))
        mAP.append(d[0])
        num_annots.append(d[1])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, losses, color="red")
    ax[0].set_ylabel("loss")
    ax[0].set_xlabel("epoch")

    ax[1].plot(epochs, mAP, color="blue")
    ax[1].set_ylabel("mAP")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylim([0.0 - 0.05, 1.0 + 0.05])

    fig.suptitle("Number of Annotated Saffrons = {0}".format(num_annots[0]))
    fig.show()


plot_loss_mAP(json_file_path="/mnt/2tra/saeedi/results/history.json")