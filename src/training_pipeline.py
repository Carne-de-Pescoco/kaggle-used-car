import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from src.pipeline import pipe_categorical
import src.config as config

data_path = "../data/raw/"

def run_training():
    """Train the model."""

    # read training data
    data = pd.read_csv(config.DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1), data[config.TARGET], test_size=0.1, random_state=config.RANDOM_STATE)  # we are setting the seed here

    pipe_categorical.fit(X_train)
    # TODO: É NECESSÁRIO COLOCAR TODAS AS TRANSFORMAÇÕES EM APENAS UM PIPELINE.
    joblib.dump(pipe_categorical, config.PIPELINE_NAME)


if __name__ == '__main__':
    run_training()
