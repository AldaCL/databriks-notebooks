import os

import pandas as pd


def get_dataframe(filename: str) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file or from DBFS, depending on the presence of the DATABRICKS_RUNTIME_VERSION
    environment variable.

    This function checks if the DATABRICKS_RUNTIME_VERSION environment variable is set. If it is set, it loads the
    DataFrame from the DBFS path; otherwise, it loads the DataFrame from the local datasource path.
    """

    dbfs_usr_path = "/dbfs/FileStore/shared_uploads/aldair.alda27@gmail.com"
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        path = f"{dbfs_usr_path}/{filename}.csv"
    else:
        path = f"datasource/{filename}.csv"

    return pd.read_csv(path)
