import pandas as pd


def convert_timestamp(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    return df


def check_missing_values(df):
    return df.isnull().sum()


def check_duplicates(df):
    return df.duplicated().sum()


def check_invalid_logins(df):
    invalid = df[
        df["failed_logins"] > df["logins"]
    ]

    return invalid


def check_negative_values(df):

    numeric_columns = [
        "logins",
        "failed_logins",
        "distinct_ips",
        "distinct_hosts",
        "mb_out",
        "dns_queries"
    ]

    results = {}

    for col in numeric_columns:
        results[col] = (df[col] < 0).sum()

    return results