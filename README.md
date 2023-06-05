Market Prediction
==============================

Regardless of your investment strategy, fluctuations are expected in the financial market. Despite this variance, professional investors try to estimate their overall returns. Risks and returns differ based on investment types and other factors, which impact stability and volatility. To attempt to predict returns, there are many computer-based algorithms and models for financial market trading. Yet, with new techniques and approaches, data science could improve quantitative researchers' ability to **forecast an investment's return**.

Task
------------
The task is to build a model that forecasts an investment's return rate.

Data
------------

This [dataset](https://www.kaggle.com/competitions/ubiquant-market-prediction/data) contains features derived from real historic data from thousands of investments. Dataset description:

`row_id` - A unique identifier for the row.

`time_id` - The ID code for the time the data was gathered. The time IDs are in order, but the real time between the time IDs is not constant and will likely be shorter for the final private test set than in the training set.

`investment_id` - The ID code for an investment. Not all investment have data in all time IDs.

`target` - The target.

`[f_0:f_299]` - Anonymized features generated from market data.

Project Organization
------------

    ├── LICENSE
    ├── README.md              <- The top-level README for developers using this project.
    ├── models                 <- Serialized models
    ├── scripts                <- .sh scripts for the fast .py scripts running
    ├── requirements.txt       <- The requirements file for reproducing the analysis environment
    ├── requirements-dev.txt   <- The requirements file for github actions 
    ├── hft/src                <- Source code for use in this project
    |    ├── data              <- Scripts to preprocess data
    |    │     └── make_dataset.py
    |    └── models            <- Scripts to train model and do inference
    |          └── train_model.py.py
    ├── dvc.yaml               <- DVC pipeline config
    └── params.yaml            <- Config file

--------

How to run
------------

- First of all, you need to install all dependencies with 
```
pip install -r requirements.txt
```
- Prepare training dataset and configure [remote DVC storage](https://dvc.org/doc/user-guide/data-management/remote-storage). The current version uses [Google Drive](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive) storage
- You can specify training parameters before the run in the [configuration file](https://github.com/pacifikus/HFT/blob/main/params.yaml)
- Run MLFlow server with your [preferred configuration](https://mlflow.org/docs/latest/tracking.html#concepts) and set the environment variable `MLFLOW_TRACKING_URI`. You can find MLFlow tracking logic in [`mlflow_utils.py`](https://github.com/pacifikus/HFT/blob/main/hft/src/models/mlflow_utils.py)
- Finally, you can run the project either with CLI or with DVC pipeline

### CLI

You can run the project with [.sh file](https://github.com/pacifikus/HFT/blob/main/scripts/start.sh):

```
scripts/start.sh
```

### DVC pipeline

Also, you can run project with DVC pipeline organized as an experiment:

```
dvc exp run
```

Pipeline stages are listed in [DVC config file](https://github.com/pacifikus/HFT/blob/main/dvc.yaml)

