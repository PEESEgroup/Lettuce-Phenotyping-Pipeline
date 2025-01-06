import pandas as pd

def load_and_prepare_data(file_path, timestamp_column, sample_prefix):
    df = pd.read_csv(file_path)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df.set_index(timestamp_column, inplace=True)

    sample_columns = [col for col in df.columns if col.startswith(sample_prefix)]

    return df, sample_columns
