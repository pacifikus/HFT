import logging
import os
import pickle

import click
import numpy as np
import pandas as pd
from box import ConfigBox
from lightgbm import LGBMRegressor
from ruamel.yaml import YAML
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

    for train_index, test_index in skf.split(train, train["investment_id"]):
        train_data = train.iloc[train_index]
        valid_data = train.iloc[test_index]

        lgbm = LGBMRegressor(
            num_leaves=2 ** np.random.randint(3, 8),
            learning_rate=10 ** (-np.random.uniform(0.1, 2)),
            n_estimators=1000,
            min_child_samples=1000,
            subsample=np.random.uniform(0.5, 1.0),
            subsample_freq=1,
            n_jobs=-1,
        )

        lgbm.fit(
            train_data[features],
            train_data[params.train.target],
            eval_set=(valid_data[features], valid_data[params.train.target]),
            early_stopping_rounds=params.train.early_stopping_rounds,
        )
        models.append(lgbm)
    save_models(models, models_filepath)
    logger.info("finish training LGBM models")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
