import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
from datetime import date

HOLIDAYS = USFederalHolidayCalendar().holidays('2011-01-01', '2016-06-30')

class Preprocessor:
    def __init__(self, drop_na=True):
        self.drop_na = drop_na

    def load_data(self, data_file) -> pd.DataFrame:
        return pd.read_csv(data_file, index_col=0, parse_dates=True)

    def get_day_value(self, df):
        """Preprocessing function. Creates day column."""
        return df.assign(**{'day': df.index.day})

    def is_holiday(self, df):
        """Return a new column is_holiday."""
        return df.assign(is_holiday=pd.to_datetime(df.index.date).isin(HOLIDAYS))

    def get_season(self, df):
        """Return the season based off:
        Dec, Jan, Feb = winter
        Mar, Apr, May = spring
        Jun, Jul, Aug = summer
        Sep, Oct, Nov = autumn
        """

        season_mapping = {4: 'winter',
                        1: 'spring',
                        2: 'summer',
                        3: 'autumn'}

        # map the dates quarter to what season it is
        offset_months = df.index - pd.DateOffset(months=1)

        return df.assign(season=offset_months.quarter.map(season_mapping))

    def remove_first(self, df, col_nam, timedelta):
        """ Return a cleaned version of column col_nam

        Input: dataframe (df) with a column (col_nam) and time window (timedelta)
        Keep the original column value if sufficient period of time has elapsed (timedelta)
        Otherwise return NaN
        """
        return (df
                .assign(**{col_nam:
                        lambda df: np.where(df.index >= (df.index.min() + pd.Timedelta(timedelta)),
                                            df[col_nam],
                                            np.nan)
                        })
            )

    def get_mean_of_previous(self, df, col_nam, timedelta,
                            prev_col='store_sales', agg='mean'):
        """Return a new column col_name with the mean count of a previous time period

        Input: dataframe (df) with a column (col_nam),
        window for rolling averages (timedelta), period of time to groupby (grouper),
        column to aggregate on and aggregation function (default 'mean').
        """

        return (
            df
            .assign(**{col_nam: (lambda df: df
                                    .sort_index()
                                        #allow for not grouping by time
                                    .assign(NoGroup=1)
                                        #rolling average using only past data
                                    .rolling(timedelta, closed='left')
                                        #aggregate on the column
                                    [prev_col].agg(agg)
                                    )})
            .pipe(self.remove_first, col_nam, timedelta)
        )

    def get_last_period(self, df, col_name, periods):
        return df.assign(**{col_name: (lambda df: df.store_sales.shift(periods))})

    def get_day_info(self, df):
        return (
            df
            .pipe(self.get_day_value)
            .pipe(self.is_holiday)
            .pipe(self.get_season)
            .assign(is_weekday=lambda df: df.wday > 2)
            .assign(is_workday=lambda df: (df['is_weekday'] == True) & (df['is_holiday'] == False))
        )

    def get_historical(self, df):
        return (
            df
            .pipe(self.get_mean_of_previous, col_nam='last_7_days', timedelta='7D')
            .pipe(self.get_mean_of_previous, col_nam='last_4_weeks', timedelta='28D')
            .pipe(self.get_last_period, col_name='yesterday', periods=1)
            .pipe(self.get_last_period, col_name='last_week', periods=7)
            .pipe(self.get_last_period, col_name='last_month', periods=30)
            .pipe(self.get_last_period, col_name='last_year', periods=365)
        )

    def preprocess(self, df):
        return (
            df
            .pipe(self.get_day_info)
            .pipe(self.get_historical)
        )

    def load_and_preprocess(self, data_file, features_to_drop=None):
        df = self.load_data(data_file)
        df_preprocessed = self.preprocess(df)
        df_preprocessed.drop(columns=features_to_drop, inplace=True)
        if self.drop_na:
            return df_preprocessed.dropna()
        return df_preprocessed

preprocessor = Preprocessor()
