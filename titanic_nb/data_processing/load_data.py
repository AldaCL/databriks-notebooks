import pandas as pd
import os


def get_dataframe(filename: str) -> pd.DataFrame:
    """
    Load dataframes from a csv file or from dbfs,
    depending on the os.environ
    """

    dbfs_usr_path = "/dbfs/FileStore/shared_uploads/aldair.alda27@gmail.com"

    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        path = f"{dbfs_usr_path}/{filename}.csv"
    else:
        path = f"datasource/{filename}.csv"

    return pd.read_csv(path)