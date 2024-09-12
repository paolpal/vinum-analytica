from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.models import TreeModel # type: ignore

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
            "criterion": "gini",
            "min_impurity_decrease": 1e-08,
            "max_depth": 1000
        }

model = TreeModel(vectorizer=vec, **hyperparams)

logging.info('Training model...')
model.train(train)
logging.info('done')

logging.info('Saving model...')
model.save('./models/tree_model.pkl')
logging.info('done')