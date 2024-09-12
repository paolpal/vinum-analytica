from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.models import TreeModel # type: ignore

import logging

logging.basicConfig(level=logging.INFO)

# Carica il dataset
logging.info('Loading dataset...')
dataset = WineDatasetManager()
dataset.load('./data/processed/test.csv')
logging.info('done')

test = dataset

model = TreeModel.load('./models/tree_model.pkl')

logging.info('Evaluating model...')
accuracy = model.evaluate(test)
logging.info('done')

logging.info(f'Accuracy: {accuracy}')

