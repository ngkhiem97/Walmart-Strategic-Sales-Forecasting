import pandas as pd

def get_day_value(df):
    """ Preprocessing function
    Creates day column.
    """
    df = df.assign(**{'day': df.index.day})
    return df

def pre_modeling(df):
    """
    Pre-modeling function for data preprocessing.
    """
    df_processed = df.pipe(get_day_value)

    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    df_processed = (df_processed.join([pd.get_dummies(df[col], prefix=col) for col in event_cols])
                            .drop(event_cols, axis=1)
                            .rename(str.lower, axis=1))

    df_processed = df_processed.dropna()
    return df_processed