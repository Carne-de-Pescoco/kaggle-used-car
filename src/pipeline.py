from feature_engine.encoding import RareLabelEncoder, CountFrequencyEncoder
from sklearn.pipeline import Pipeline

import src.config as config

pipe_categorical = Pipeline(
    [
        ('RareLabel', RareLabelEncoder(tol=0.05, n_categories=4, variables=config.CAR_BRAND, replace_with='Rare')),
        ('CountFrequency', CountFrequencyEncoder(encoding_method='frequency', variables=config.CAR_BRAND))
    ]
)