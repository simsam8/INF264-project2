import numpy as np
from ModelOptimization import ModelOptimization
from sklearn.model_selection import train_test_split
from DataPreprocessing import DataPreprocessing

seed = 100

dpp = DataPreprocessing()

X = np.load("emnist_hex_images.npy")
y = np.load("emnist_hex_labels.npy")


processed_X = dpp.process_data(X)


X_train, X_test, y_train, y_test = train_test_split(
    processed_X, y, test_size=0.2, train_size=0.8, random_state=seed
)

mo: ModelOptimization = ModelOptimization(random_state=seed)
mo.train_and_save_best_models(X_train, y_train)
