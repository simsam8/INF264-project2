from loguru import logger
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from time import time
import pickle


class ModelOptimization:
    """
    Class for training and running cross validation on different models
    """

    def __init__(self, random_state=None) -> None:
        """
        Sets a random state and StratifiedKFold

        Parameters:
        ----------
        random_state: provides a seed for all methods that use random_state
        """
        self.random_state = random_state
        self.skf = StratifiedKFold(
            n_splits=5, random_state=self.random_state, shuffle=True
        )

    def cross_validate(self, X, y, estimator, parameters: dict) -> tuple:
        """
        Runs cross validation with the same StratifiedKFold for
        all models and train the given model on the whole training data.

        Parameters:
        ----------
        X: feature
        y: labels
        estimator: model
        parameters: parameters used in cross validation

        return: trained model,
                model score,
                mean fit time,
                mean prediction time,
                total cv time
        """
        gscv = GridSearchCV(estimator, parameters, cv=self.skf, verbose=3, n_jobs=7)
        start_time = time()
        gscv.fit(X, y)
        end_time = time()

        total_validation_time = end_time - start_time
        best_model = gscv.best_estimator_
        best_score = gscv.best_score_
        best_params = gscv.best_params_
        best_params_index = gscv.cv_results_["params"].index(best_params)
        mean_fit_time = gscv.cv_results_["mean_fit_time"][best_params_index]
        mean_pred_time = gscv.cv_results_["mean_score_time"][best_params_index]

        print(f"Mean score of best estimator: {best_score}")
        print(f"With parameters: {best_params}\n")
        print(f"Mean fit time: {mean_fit_time}")
        print(f"Mean prediction time: {mean_pred_time}")
        print(f"Total cross validation time: {total_validation_time}")
        return (
            best_model,
            best_score,
            mean_fit_time,
            mean_pred_time,
            total_validation_time,
        )

    def train_and_save_best_models(self, X, y) -> None:
        """
        Method for training and saving models

        Parameters:
        ----------
        X: features
        y: labels
        """
        models = dict()

        models["SVC"] = self.support_vector(X, y)
        models["KNN"] = self.k_neighbors(X, y)
        models["RandomForest"] = self.random_forest(X, y)
        models["MLPC"] = self.multi_layer_perceptron(X, y)

        with open("models/trained_models.pickle", "wb") as f:
            pickle.dump(models, file=f)

    def load_trained_models(self) -> dict:
        """
        Loads trained models from file

        Use the model name as key

        available models:

        SVC - Support Vector Classifier
        KNN - K_Nearest Neigbors Classifier
        RandomForest - Random Forest Classifier
        MLPC - Multi layer perceptron Classifier

        return: dictionary of models
        """
        with open("models/trained_models.pickle", "rb") as f:
            models = pickle.load(f)

        return models

    def support_vector(self, X, y) -> tuple[SVC, float, float, float]:
        """
        Estimated time on full training set: 3 hours
        """
        parameters = {
            "kernel": ["rbf"],
            "gamma": [0.1, 0.01, 0.001, 0.0001],
            "C": [1, 10, 100, 1000],
        }

        logger.info("Running parameter testing for support vector")

        svc = SVC(random_state=self.random_state)
        return self.cross_validate(X, y, svc, parameters)

    def random_forest(self, X, y) -> tuple[RandomForestClassifier, float, float, float]:
        parameters = {
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2"],
            "min_samples_split": [2, 6, 10],
        }

        logger.info("Running parameter testing for random forest")

        r_forest = RandomForestClassifier(random_state=self.random_state)
        return self.cross_validate(X, y, r_forest, parameters)

    def k_neighbors(self, X, y) -> tuple[KNeighborsClassifier, float, float, float]:
        parameters = {
            "n_neighbors": list(range(1, 22, 2)),
            "weights": ["uniform", "distance"],
            "metric": ["manhattan", "minkowski", "euclidean"],
        }

        logger.info("Running parameter testing for k_neighbors")

        knn = KNeighborsClassifier()

        return self.cross_validate(X, y, knn, parameters)

    def multi_layer_perceptron(self, X, y) -> tuple[MLPClassifier, float, float, float]:
        parameters = {
            "hidden_layer_sizes": [
                (50,),
                (100,),
                (200,),
                (300,),
                (50, 10),
                (100, 20),
                (200, 40),
                (300, 80),
            ],
            "activation": ["tanh", "relu"],
            "alpha": [0.0001, 0.001, 0.01],
        }

        logger.info("Running parameter testing for multi_layer_perceptron")

        mlpc = MLPClassifier(random_state=self.random_state, max_iter=800)
        return self.cross_validate(X, y, mlpc, parameters)
