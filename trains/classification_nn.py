from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.models import NeuralNetworkModel # type: ignore
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

best_neural = next((model for model in best if model['model_name'] == 'nn'), None)
if best_neural:
    hyperparams = best_neural['hyperparams']
else:
    logging.warning('No best model found with model_name "dt"')
    exit()

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