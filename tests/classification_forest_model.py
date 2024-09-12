from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.models import RandomForestModel # type: ignore

import logging

logging.basicConfig(level=logging.INFO)

# Carica il dataset
logging.info('Loading dataset...')
dataset = WineDatasetManager()
dataset.load('./data/processed/test.csv')
logging.info('done')

test = dataset

model = RandomForestModel.load('./models/random_forest_model.pkl')

logging.info('Evaluating model...')
accuracy = model.evaluate(test)
logging.info('done')

logging.info(f'Accuracy: {accuracy}')