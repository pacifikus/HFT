# -*- coding: utf-8 -*-
import logging

import click
import pandas as pd


def read_data(input_filepath):
    return pd.read_parquet(input_filepath, engine="pyarrow")


def drop_outliers(train_data):
    outlier_list = []
    outlier_col = []

    for col in (f"f_{i}" for i in range(300)):
        _mean, _std = train_data[col].mean(), train_data[col].std()

        temp_df = train_data.loc[
            (train_data[col] > _mean + _std * 70)
            | (train_data[col] < _mean - _std * 70)
        ]
        temp2_df = train_data.loc[
            (train_data[col] > _mean + _std * 35)
            | (train_data[col] < _mean - _std * 35)
        ]
        if len(temp_df) > 0:
            outliers = temp_df.index.to_list()
            outlier_list.extend(outliers)
            outlier_col.append(col)

        elif 0 < len(temp2_df) < 6:
            outliers = temp2_df.index.to_list()
            outlier_list.extend(outliers)
            outlier_col.append(col)

    outlier_list = list(set(outlier_list))
    train_data = train_data.drop(train_data.index[outlier_list])
    return train_data


@click.command()
@click.option(
    "-i",
    "--input_filepath",
    default='data/raw/train_data.parquet',
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output_filepath",
    default='data/processed/result.parquet',
    type=click.Path(),
)
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    train_data = read_data(input_filepath)
    train_data = drop_outliers(train_data)
    train_data.to_parquet(output_filepath, index=False, engine="pyarrow")
    logger.info(f"final data were saved to {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
