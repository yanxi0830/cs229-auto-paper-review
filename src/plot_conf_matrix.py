from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    binary1 = np.array([[154, 18],
                        [36, 45]])

    plt.rcParams.update({'font.size': 26})
    plt.rcParams.update({"figure.figsize": [6.4, 6.4]})

    fig1, ax1 = plot_confusion_matrix(conf_mat=binary1,
                                      show_absolute=True,
                                      show_normed=True,
                                      colorbar=True)

    fig1.savefig('conf_mat.pdf')
