from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.models import TreeModel # type: ignore
import json
import logging

logging.basicConfig(level=logging.INFO)

# Carica il dataset
logging.info('Loading dataset...')
dataset = WineDatasetManager()
dataset.load('./data/processed/train_classification.csv')
logging.info('done')

train = dataset
vec = train.vectorize()
train.resample()

with open('./results/best.json', 'r') as f:
    best = json.load(f)

best_tree = next((model for model in best if model['model_name'] == 'dt'), None)
if best_tree:
    hyperparams = best_tree['hyperparams']
else:
    logging.warning('No best model found with model_name "dt"')
    exit()

model = TreeModel(vectorizer=vec, **hyperparams)

logging.info('Training model...')
model.train(train)
logging.info('done')

logging.info('Saving model...')
model.save('./models/tree_model.pkl')
logging.info('done')