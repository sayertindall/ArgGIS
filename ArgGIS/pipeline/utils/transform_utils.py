import pandas as pd

def detect_header_row(df, start=2, end=10, min_nonnull=3):
    for i in range(start, end):
        if df.iloc[i].notnull().sum() > min_nonnull:
            return i
    return end

def normalize_columns(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return df

def tidy_reserve_table(df):
    id_cols = df.columns[:5]  # operator, basin, province, etc.
    value_cols = df.columns[5:]
    df_melted = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="metric", value_name="value")
    return df_melted
