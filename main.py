from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.models import TreeModel, RandomForestModel, NeuralNetworkModel, RegressorTreeModel# type: ignore
from vinum_analytica.visualization import Plotter # type: ignore

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix

#dataset = WineDatasetManager.read_csv('./data/raw/winemag-data-130k-v2.csv')
# Preprocessa il dataset
#dataset.preprocess()

print('Loading dataset...', end='')
dataset = WineDatasetManager(y_label='price')
#dataset.load('./data/processed/train.csv')
dataset.load('./data/processed/train_regression.csv')
print('done')

#train, valid = dataset.split()
train = dataset
vec = train.vectorize()

#print('Oversampling dataset...', end='')
#train.oversample()
#print('done')

# Inizializza il modello
#forest = RandomForestModel(vectorizer=vec)
"""nn = NeuralNetworkModel(
        input_size=vec.get_feature_names_out().shape[0], 
        output_size=len(train.classes()),
        epochs=1,
        vectorizer=vec
    )
"""
#tree = TreeModel(vectorizer=vec)
tree = RegressorTreeModel(vectorizer=vec)

# Addestra il modello
print('Training model...', end='')
#forest.train(train)
tree.train(train)
#nn.train(train)
print('done')

# Valuta il modello
#print('Evaluating model...', end='')
#accuracy = tree.evaluate(valid, mean_squared_error)
#accuracy = forest.evaluate(valid)
#accuracy = nn.evaluate(valid)
#print('done')
#print(f'Accuracy: {accuracy}')

#forest.save('./models/forest_model_f.pkl')
#nn.save('./models/nn_model.pkl'
tree.save('./models/regressor_tree_model.pkl')

#print('Computing confusion matrix...', end='')
#cm = tree.evaluate(valid, metric=confusion_matrix, normalize='true')
#cm = forest.evaluate(valid, metric=confusion_matrix, normalize='true')
#cm = nn.evaluate(valid, metric=confusion_matrix, normalize='true')
#print('done')
#plotter = Plotter()

#plotter.plot_confusion_matrix(cm, classes=tree.classes())