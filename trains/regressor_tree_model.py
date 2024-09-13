from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.models import RegressorTreeModel # type: ignore

import logging

logging.basicConfig(level=logging.INFO)

# Carica il dataset
logging.info('Loading dataset...')
dataset = WineDatasetManager(y_label='price')
dataset.load('./data/processed/train_regression.csv')
logging.info('done')

train = dataset
vec = train.vectorize()

model = RegressorTreeModel(
        vectorizer=vec
    )

logging.info('Training model...')
model.train(train)
logging.info('done')

logging.info('Saving model...')
model.save('./models/regressor_tree_model.pkl')
logging.info('done')