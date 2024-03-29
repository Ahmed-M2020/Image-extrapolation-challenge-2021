import os
from matplotlib import pyplot as plt


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets, and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(2, 2)
    ax[1, 1].remove()

    for i in range(len(inputs)):
        ax[0, 0].clear()
        ax[0, 0].set_title('input')
        ax[0, 0].imshow(inputs[i, 0], cmap=plt.cm.gray, interpolation='none')
        ax[0, 0].set_axis_off()
        ax[0, 1].clear()
        ax[0, 1].set_title('targets')
        ax[0, 1].imshow(targets[i, 0], cmap=plt.cm.gray, interpolation='none')
        ax[0, 1].set_axis_off()
        ax[1, 0].clear()
        ax[1, 0].set_title('predictions')
        ax[1, 0].imshow(predictions[i, 0], cmap=plt.cm.gray, interpolation='none')
        ax[1, 0].set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=1000)
    del fig
    