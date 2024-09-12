from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.visualization import Plotter # type: ignore

import logging

logging.basicConfig(level=logging.INFO)

# Carica il dataset
logging.info('Loading dataset...')
dataset = WineDatasetManager()
dataset.load('./data/processed/train_classification.csv')
logging.info('done')

dataset.vectorize()
dataset.resample()

plotter = Plotter()
plotter.plot_variety_count(dataset)