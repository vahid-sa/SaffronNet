import numpy as np
import matplotlib.pyplot as plt
from retinanet.settings import MAX_ANOT_ANCHOR_POSITION_DISTANCE
Dxy_zarib = 10.0
Dalpha_zarib = 1.0
ACC_ZARIB = 11.0
IGNORE_ZARIB = 13.0


def alpha(x):
    return ((-Dxy_zarib * x) - (-ACC_ZARIB * MAX_ANOT_ANCHOR_POSITION_DISTANCE)) / Dalpha_zarib


def alpha_ignore(x):
    return ((-Dxy_zarib * x) - (-IGNORE_ZARIB * MAX_ANOT_ANCHOR_POSITION_DISTANCE)) / Dalpha_zarib


def plot():
    dxy = np.arange(0, 12, 0.01)
    dalpha = alpha(dxy)
    dalpha_ignore = alpha_ignore(dxy)

    fig, ax = plt.subplots()
    ax.plot(dxy, dalpha, color='green', label='accept')

    ax.plot(dxy, dalpha_ignore, color='orange', label="ignore")
    plt.legend()

    ax.set(xlabel='position distance (px)', ylabel='angle distance (dergree)',
           title='Anchor acceptance as groundtruth')
    plt.xlim(xmin=0)  # this line
    plt.ylim(ymin=0)  # this line
    plt.grid()
    plt.fill_between(dxy, dalpha, color='green')
    plt.fill_between(dxy, dalpha, dalpha_ignore, color='orange')

    plt.show()


plot()
