import sys
import time

from vinum_analytica.data.dataset import WineDatasetManager # type: ignore
from vinum_analytica.models.tree_model import TreeModel # type: ignore
from vinum_analytica.models.forest_model import RandomForestModel # type: ignore
from vinum_analytica.models.nn_model import NeuralNetworkModel # type: ignore
from vinum_analytica.visualization.plots import Plotter # type: ignore

from sklearn.metrics import confusion_matrix

#dataset = WineDatasetManager.read_csv('./data/raw/winemag-data-130k-v2.csv')
# Preprocessa il dataset
#dataset.preprocess()

dataset = WineDatasetManager()
dataset.load('./data/processed/train.csv')


train, valid = dataset.split()
vec = train.vectorize()
train.oversample()

# Inizializza il modello
#forest = RandomForestModel()
nn = NeuralNetworkModel(
        input_size=vec.get_feature_names_out().shape[0], 
        output_size=len(train.classes()), 
        vectorizer=vec
    )

# Addestra il modello
#forest.train(train)
#forest.load('./models/forest_model_f.pkl')

nn.train(train)

# Valuta il modello
#accuracy = forest.evaluate(valid)
accuracy = nn.evaluate(valid)
print(f'Accuracy: {accuracy}')

#forest.save('./models/forest_model_f.pkl')
nn.save('./models/nn_model.pkl')

#cm = forest.evaluate(valid, metric=confusion_matrix, normalize='true')
cm = nn.evaluate(valid, metric=confusion_matrix, normalize='true')

plotter = Plotter()

plotter.plot_confusion_matrix(cm, classes=nn.classes())