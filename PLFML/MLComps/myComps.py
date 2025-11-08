
# custom_components.py

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from plf.utils import Component


# ------------------------
# DataLoaderComponent
# ------------------------
class DataLoaderComponent(Component):
    def __init__(self):
        super().__init__()
        # Declare arguments users can customize
        self.args = {
            "shuffle",       # Whether to shuffle the data
            "sample_size",   # How many samples to take
            "feature_indices"  # Which columns/features to select
        }

    def _setup(self, args):
        self.shuffle = args.get("shuffle", True)
        self.sample_size = args.get("sample_size", None)
        self.feature_indices = args.get("feature_indices", None)

    def run(self):
        X, y = load_diabetes(return_X_y=True)

        # Optionally select only certain features
        if self.feature_indices is not None:
            X = X[:, self.feature_indices]

        # Optionally shuffle
        if self.shuffle:
            idx = np.random.permutation(len(y))
            X, y = X[idx], y[idx]

        # Optionally take a subset
        if self.sample_size:
            X, y = X[:self.sample_size], y[:self.sample_size]

        return X, y


# ------------------------
# PreprocessorComponent
# ------------------------
class PreprocessorComponent(Component):
    def __init__(self):
        super().__init__()
        self.args = {"scaler_type", "log_transform"}

    def _setup(self, args):
        self.scaler_type = args.get("scaler_type", "minmax")
        self.log_transform = args.get("log_transform", False)

    def run(self, X):
        # Apply log transform if needed
        if self.log_transform:
            X = np.log1p(X)

        # Select scaler
        if self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif self.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.scaler_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler


# ------------------------
# RegressorComponent
# ------------------------
class RegressorComponent(Component):
    def __init__(self):
        super().__init__()
        self.args = {"model_type", "model_params"}

    def _setup(self, args):
        self.model_type = args.get("model_type", "linear")
        self.model_params = args.get("model_params", {})

    def run(self, X, y):
        # Choose model
        if self.model_type == "linear":
            model = LinearRegression(**self.model_params)
        elif self.model_type == "ridge":
            model = Ridge(**self.model_params)
        elif self.model_type == "lasso":
            model = Lasso(**self.model_params)
        elif self.model_type == "rf":
            model = RandomForestRegressor(**self.model_params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        model.fit(X, y)
        return model


# ------------------------
# EvaluatorComponent
# ------------------------
class EvaluatorComponent(Component):
    def __init__(self):
        super().__init__()
        self.args = {"metrics"}

    def _setup(self, args):
        # Accept list of metrics or default to r2 + rmse
        self.metrics = args.get("metrics", ["r2", "rmse"])

    def run(self, model, X, y):
        y_pred = model.predict(X)
        results = {}
        for metric in self.metrics:
            if metric == "r2":
                results["r2"] = r2_score(y, y_pred)
            elif metric == "rmse":
                results["rmse"] = np.sqrt(mean_squared_error(y, y_pred))
            elif metric == "mae":
                results["mae"] = mean_absolute_error(y, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return results



from plf.utils import WorkFlow
import os, json
from copy import deepcopy
from pathlib import Path
import joblib  # For saving models

class SimpleRegressionWorkflow(WorkFlow):
    def __init__(self):
        super().__init__()
        self.template = {'loader', 'preprocessor', 'regressor', 'evaluator'}
        self.paths = {'model', 'metrics'}
    def _setup(self, args):
        pass

    def new(self, args):
        # Ensure all required configuration keys are provided
        if not self.template.issubset(set(args.keys())):
            raise ValueError(f'the args should have {", ".join(self.template- set(list(args.keys())))}')
        
    def prepare(self):
        args = deepcopy(self.P.cnfg['args'])
        self.loader = self.load_component(**args["loader"])
        self.preprocessor = self.load_component(**args["preprocessor"])
        self.regressor = self.load_component(**args["regressor"])
        self.evaluator = self.load_component(**args["evaluator"])
        return True

    def run(self):
        X, y = self.loader.run()
        X_scaled, scaler = self.preprocessor.run(X)
        model = self.regressor.run(X_scaled, y)
        results = self.evaluator.run(model, X_scaled, y)
        # Save outputs
        with open(self.P.get_path(of="metrics"), "w") as f:
            json.dump(results, f)
        print("Metrics:", results)
        
        # 6. Save model
        model_path = self.P.get_path(of = "model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved at: {model_path}")

    def get_path(self, of, pplid, args):
        if of == "metrics":
            path = Path('metrics') / f"{pplid}.json"
        elif of == "model":
            path = Path('models') /  f"{pplid}.pkl"
        else:
            raise NotImplementedError

        return path
    
    
    def status(self):
        import json
        with open(self.P.get_path(of='metrics')) as fl:
            data = json.load(fl)
        return data
