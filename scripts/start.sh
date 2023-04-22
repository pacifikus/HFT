python hft/src/data/make_dataset.py -i "data/raw/train_data.parquet" -o "data/processed/result.parquet"
python hft/src/models/train_model.py -i "data/processed/result.parquet" -o "models"