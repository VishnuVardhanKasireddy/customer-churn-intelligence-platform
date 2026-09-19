import pandas as pd
from config.settings import (TRAIN_DATA_FILE,VALIDATION_DATA_FILE,TEST_DATA_FILE)
from src.model.preprocessing import prepare_data

def load_model_data():
    train = pd.read_csv(TRAIN_DATA_FILE)
    validation = pd.read_csv(VALIDATION_DATA_FILE)
    test = pd.read_csv(TEST_DATA_FILE)

    return prepare_data(train,validation,test)

