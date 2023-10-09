import numpy as np


class DataPreprocessing:
    def __init__(self) -> None:
        pass

    def linear_scaling(self, data):
        min_val = np.min(data)
        max_val = np.max(data)

        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data

    def filter_color_values(self, data):
        # 35 65
        black_filter = np.where(data < 35, 0, data)
        white_filter = np.where(black_filter > 250, 255, black_filter)
        # data[data < 100] = 0
        # data[data > 250] = 255
        return white_filter

    def smoothing(self, data, kernel_size=10):
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved = np.convolve(data, kernel, mode="same")
        return data_convolved

    def process_data(self, data):
        color_filter = self.filter_color_values(data)
        normalize = self.linear_scaling(color_filter)
        return normalize
