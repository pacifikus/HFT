import json
import os
import pickle

import mlflow
import pandas as pd
from box import ConfigBox
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def log_to_mlflow():
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(params.base.exp_name)

    with mlflow.start_run():
        raw_metrics_df = pd.read_csv(params.base.raw_metrics_path)
        for metric, value in raw_metrics_df.mean().to_dict().items():
            mlflow.log_metric(
                key=metric,
                value=value,
            )

        mlflow.log_params(
            {
                "seed": params.train.seed,
                "early_stopping_rounds": params.train.early_stopping_rounds,
                "estimator_params": json.dumps(params.train.lbgm),
            }
        )

        count = 1
        for model_path in os.listdir("models"):
            if model_path.endswith(".pkl"):
                with open(f"models/{model_path}", "rb") as fb:
                    model = pickle.load(fb)
                mlflow.lightgbm.log_model(model, f"model_{count}")
                count += 1
