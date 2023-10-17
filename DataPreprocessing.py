import numpy as np


class DataPreprocessing:
    """
    Class for preprocessing image data
    """
    def __init__(self) -> None:
        pass

    def linear_scaling(self, data: np.ndarray) -> np.ndarray:
        """
        Scale pixel values between 0 and 1

        Parameters:
        ----------
        data: nd array of greyscale pixel values

        return: scaled data
        """
        min_val = np.min(data)
        max_val = np.max(data)

        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data

    def filter_color_values(
        self, data: np.ndarray, black_cutoff=35, white_cutoff=250
    ) -> np.ndarray:
        """
        Filters pixel values with given cutoffs

        Parameters:
        ----------
        data: nd array of greyscale pixel values
        black_cutoff: pixels under cutoff set to 0
        white_cutoff: pixels over cutoff set to 255

        return: filtered data
        """
        black_filter = np.where(data < black_cutoff, 0, data)
        white_filter = np.where(black_filter > white_cutoff, 255, black_filter)
        return white_filter

    def process_data(self, data: np.ndarray) -> np.ndarray:
        """
        applies linear scaling and color filter

        Parameters:
        ----------
        data: nd array of greyscale pixel values

        return: processed data
        """
        color_filter = self.filter_color_values(data)
        normalize = self.linear_scaling(color_filter)
        return normalize
