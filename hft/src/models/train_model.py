import click
import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
import pickle

seed = 8
folds = 5
target = 'target'


def save_models(models, models_filepath):
    for i, model in enumerate(models):
        with open(f'{models_filepath}/lgbm_{i}.pkl', 'wb') as f:
            pickle.dump(model, f)


@click.command()
@click.option('-i', '--input_filepath', type=click.Path(exists=True))
@click.option('-o', '--models_filepath', type=click.Path())
def main(input_filepath, models_filepath):
    logger = logging.getLogger(__name__)
    logger.info(f'start training LGBM models')

    train = pd.read_parquet(input_filepath)
    features = train.columns[4:]
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    models = []

    for train_index, test_index in skf.split(train, train['investment_id']):
        train_data = train.iloc[train_index]
        valid_data = train.iloc[test_index]

        lgbm = LGBMRegressor(
            num_leaves=2 ** np.random.randint(3, 8),
            learning_rate=10 ** (-np.random.uniform(0.1, 2)),
            n_estimators=1000,
            min_child_samples=1000,
            subsample=np.random.uniform(0.5, 1.0),
            subsample_freq=1,
            n_jobs=-1
        )

        lgbm.fit(
            train_data[features],
            train_data[target],
            eval_set=(valid_data[features], valid_data[target]),
            early_stopping_rounds=10,
        )
        models.append(lgbm)
    save_models(models, models_filepath)
    logger.info(f'finish training LGBM models')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
