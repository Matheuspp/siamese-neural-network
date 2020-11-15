import matplotlib.pyplot as plt


def plot_acc(data, show=False):
    plt.plot(data[0], data[1])
    if show:
        plt.show()
