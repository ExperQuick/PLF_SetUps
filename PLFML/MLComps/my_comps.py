from sklearn.datasets import load_diabetes
from plf.utils import Component

class DataLoaderComponent(Component):
    def __init__(self):
        super().__init__()
    def _setup(self, args):
        pass

    def run(self):
        x,y = load_diabetes(return_X_y=True)
        return x, y


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from plf.utils import Component

class PreprocessorComponent(Component):
    def __init__(self):
        super().__init__()
        self.args = {"scaler_type"}

    def _setup(self, args):
        self.scaler_type = args.get("scaler_type", "minmax")

    def run(self, X):
        if self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif self.scaler_type == "standard":
            scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler


from sklearn.linear_model import LinearRegression
from plf.utils import Component

class RegressorComponent(Component):
    def run(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model


from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from plf.utils import Component

class EvaluatorComponent(Component):
    def run(self, model, X, y):
        y_pred = model.predict(X)
        return {
            "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred))
        }


from plf.utils import WorkFlow
import os, json
from copy import deepcopy

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

    def run(self):
        X, y = self.loader.run()
        X_scaled, scaler = self.preprocessor.run(X)
        model = self.regressor.run(X_scaled, y)
        results = self.evaluator.run(model, X_scaled, y)
        # Save outputs
        with open(self.get_path("metrics", self.P.pplid, {}), "w") as f:
            json.dump(results, f)
        print("Metrics:", results)
        
        # 6. Save model
        model_path = self.get_path("model", self.P.pplid, {})
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved at: {model_path}")

    def get_path(self, of, pplid, args):
        if of == "metrics":
            path = os.path.join('metrics',f"{pplid}.json")
        elif of == "model":
            path = os.path.join('models', f"{pplid}.pkl")
        else:
            raise NotImplementedError

        return path
