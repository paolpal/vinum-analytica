from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.models import RegressorNeuralNetworkModel # type: ignore

import logging

logging.basicConfig(level=logging.INFO)

# Carica il dataset
logging.info('Loading dataset...')
dataset = WineDatasetManager(y_label='price')
dataset.load('./data/processed/train_regression.csv')
logging.info('done')

train = dataset
vec = train.vectorize()
scl = train.normalize()

hyperparams = {
            "epochs": 5,
            "lr": 0.0005
        }

model = RegressorNeuralNetworkModel(
    input_size=vec.get_feature_names_out().shape[0],
    vectorizer=vec, 
    scaler=scl, **hyperparams)

logging.info('Training model...')
model.train(train)
logging.info('done')

logging.info('Saving model...')
model.save('./models/regressor_nn_model.pkl')
logging.info('done')