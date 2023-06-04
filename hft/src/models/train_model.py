import logging
import os
import pickle

import click
import pandas as pd
from box import ConfigBox
from lightgbm import LGBMRegressor
from mlflow_utils import log_to_mlflow
from ruamel.yaml import YAML
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold

yaml = YAML(typ="safe")


def save_models(models, models_filepath):
    if not os.path.exists(models_filepath):
        os.makedirs(models_filepath)

    for i, model in enumerate(models):
        with open(f"{models_filepath}/lgbm_{i}.pkl", "wb") as f:
            pickle.dump(model, f)


@click.command()
@click.option(
    "-i",
    "--input_filepath",
    default="data/processed/result.parquet",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--models_filepath",
    default="models",
    type=click.Path(),
)
def main(input_filepath, models_filepath):
    logger = logging.getLogger(__name__)
    logger.info("start training LGBM models")

    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    train = pd.read_parquet(input_filepath)
    features = train.columns[4:]
    skf = StratifiedKFold(
        n_splits=params.train.folds,
        shuffle=True,
        random_state=params.train.seed,
    )
    models = []
    metrics = []

    for train_index, test_index in skf.split(train, train["investment_id"]):
        train_data = train.iloc[train_index]
        valid_data = train.iloc[test_index]

        lgbm = LGBMRegressor(
            num_leaves=params.train.lbgm.num_leaves,
            learning_rate=params.train.lbgm.learning_rate,
            n_estimators=params.train.lbgm.n_estimators,
            min_child_samples=params.train.lbgm.min_child_samples,
            subsample_freq=params.train.lbgm.subsample_freq,
            n_jobs=-1,
        )

        lgbm.fit(
            train_data[features],
            train_data[params.train.target],
            eval_set=(valid_data[features], valid_data[params.train.target]),
            early_stopping_rounds=params.train.early_stopping_rounds,
        )
        rmse = mean_squared_error(
            valid_data[params.train.target],
            lgbm.predict(valid_data[features]),
            squared=False,
        )
        mae = mean_absolute_error(
            valid_data[params.train.target],
            lgbm.predict(valid_data[features]),
        )
        metrics.append({"rmse": rmse, "mae": mae})
        models.append(lgbm)

    save_models(models, models_filepath)
    pd.DataFrame(metrics).to_csv(params.base.raw_metrics_path, index=False)
    logger.info("finish training LGBM models")
    log_to_mlflow()
    logger.info("Experiment results are logged to MLFlow")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
