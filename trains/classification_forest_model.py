from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.models import RandomForestModel # type: ignore

import logging

logging.basicConfig(level=logging.INFO)

# Carica il dataset
logging.info('Loading dataset...')
dataset = WineDatasetManager()
dataset.load('./data/processed/train.csv')
logging.info('done')

train = dataset
vec = train.vectorize()
train.oversample()

hyperparams = {
            "n_estimators": 100,
            "criterion": "gini",
            "min_impurity_decrease": 0.0,
            "max_depth": 1000
        }

model = RandomForestModel(vectorizer=vec, **hyperparams)

logging.info('Training model...')
model.train(train)
logging.info('done')

logging.info('Saving model...')
model.save('./models/random_forest_model.pkl')
logging.info('done')