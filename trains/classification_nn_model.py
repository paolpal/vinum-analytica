from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.models import NeuralNetworkModel # type: ignore

import logging

logging.basicConfig(level=logging.INFO)

# Carica il dataset
logging.info('Loading dataset...')
dataset = WineDatasetManager()
dataset.load('./data/processed/train.csv')
logging.info('done')

train = dataset
vec = train.vectorize()

hyperparams = {
            "hidden_size": 64,
            "epochs": 5,
            "lr": 0.001
        }


model = NeuralNetworkModel(
    input_size=vec.get_feature_names_out().shape[0],
    output_size=len(train.classes()),
    vectorizer=vec, **hyperparams)

logging.info('Training model...')
model.train(train)
logging.info('done')

logging.info('Saving model...')
model.save('./models/nn_model.pkl')
logging.info('done')