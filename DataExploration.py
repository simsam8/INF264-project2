import numpy as np
import matplotlib.pyplot as plt


class DataExploration:
    def __init__(self, features, labels) -> None:
        self.features = features
        self.labels = labels

    def display_class_distribution(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        plt.title("Class distribution")
        plt.xlabel("class label")
        plt.ylabel("class frequency")
        plt.xticks(list(range(17)))
        bars = plt.bar(unique, counts)
        plt.bar_label(bars, rotation=90, padding=3)
        plt.tight_layout()
        plt.margins(y=0.2)
        plt.show()

    def display_pixel_distribution(self):
        plt.title("Pixel distribution")
        unique, counts = np.unique(self.features, return_counts=True)
        plt.bar(unique, counts)
        plt.gca().ticklabel_format(
            axis="y", style="sci", scilimits=(0, 0), useMathText=True
        )
        plt.xlabel("pixel value")
        plt.ylabel("pixel frequency")
        plt.tight_layout()
        plt.show()

    def display_image(self, label=None, index=None):
        """
        Display a random or choosen image from the dataset.

        When index is choosen, ignores label.

        Params:
        ----------
        label: class label integer
        index: image index (ignores label)
        """
        if index is not None:
            plt.imshow(
                self.features[index].reshape(20, 20), vmin=0, vmax=255, cmap="gray"
            )
            plt.show()
            return

        if label is not None:
            label_list = np.where(self.labels == label)[0]
            index = np.random.choice(label_list, 1, False)
        else:
            index = np.random.choice(self.labels.shape[0], 1, False)

        plt.imshow(self.features[index].reshape(20, 20), vmin=0, vmax=255, cmap="gray")
        plt.show()
