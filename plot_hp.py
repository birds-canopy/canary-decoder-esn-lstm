import matplotlib.pyplot as plt
from hyperplot import plot_hyperopt_report

if __name__ == "__main__":

    fig = plot_hyperopt_report(f"hyper/8kHz-13mfcc+delta+delta2-wide-nonorm", ["iss", "isd", "isd2"],
                               metric="Accuracy", title="ESN model")

    plt.show()
    fig.savefig("figures/hp_esn.png")
