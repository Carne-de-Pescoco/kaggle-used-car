import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.utils.validation import check_is_fitted


def non_numeric_to_nan(df: pd.DataFrame, lst_numeric: list) -> pd.DataFrame:
    """ This function only transform non numeric values of numeric columns into np.nans"""

    df_transformed = df.copy()
    for feature in lst_numeric:
        if is_string_dtype(df_transformed[feature]):
            df_transformed.loc[~df_transformed[feature].str.isnumeric(), feature] = np.nan
            df_transformed[feature] = df_transformed[feature].astype('float64')
    

    return df_transformed


def get_car_brand(string):
    """
    Essa função, extrai a marca do carro. Ela extrai os caracteres anteriores ao primeiro espaço.
    Inputs: string
    Output: slice de string
    """
    index = string.find(" ")
    return string[:index]


class KNNImputerDataframe(BaseEstimator, TransformerMixin):

    def __init__(self, *, lst_numeric: list) -> None:
        if not isinstance(lst_numeric, list):
            self.lst_numeric = [lst_numeric]
        else:
            self.lst_numeric = lst_numeric


    def fit(self, X: pd.DataFrame):
        # persist mode in a dictionary
        model = KNNImputer(n_neighbors=2, weights='distance')
        model.fit(X[self.lst_numeric])
        self.model = model

        return self


    def transform(self, X_test: pd.DataFrame):
        X_train = self.X_train.copy()
        model = KNNImputer(n_neighbors=2, weights='distance')
        model.fit(X_train.loc[:, self.lst_numeric])
        df_transformed = model.transform(X_test.loc[:, self.lst_numeric])
        lst_columns = self.X_train.columns.tolist()
        lst_index = self.X_train.index.tolist()

        df_transformed = pd.DataFrame(df_transformed, columns=self.lst_numeric, index=lst_index)
        
        X_test.drop(self.lst_numeric, axis=1, inplace=True)
        df_transformed = pd.concat([X_test, df_transformed], axis=1)
        df_transformed = df_transformed[lst_columns]
        
        return df_transformed
