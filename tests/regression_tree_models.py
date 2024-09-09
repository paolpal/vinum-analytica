from vinum_analytica.data.dataset import WineDatasetManager # type: ignore
from vinum_analytica.models.tree_model import TreeModel # type: ignore
from vinum_analytica.models.forest_model import RandomForestModel # type: ignore
from vinum_analytica.models.nn_model import NeuralNetworkModel # type: ignore
from vinum_analytica.models.regressor_tree_model import RegressorTreeModel # type: ignore
from vinum_analytica.visualization.plots import Plotter # type: ignore

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix

print('Loading dataset...', end='')
dataset = WineDatasetManager(y_label='price')
dataset.load('./data/processed/test_regression.csv')
print('done')

tree = RegressorTreeModel.load('./models/regressor_tree_model.pkl')

# Valuta il modello
print('Evaluating model...', end='')
mse = tree.evaluate(dataset, mean_squared_error)
print('done')

print(f'Mean Squared Error: {mse}')
print()

X, y = dataset.get_x(), dataset.get_y()
# Esegui una predizione
print('Predicting...', end='')
y_pred = tree.predict(X)
print('done')

mse = 0
me = 0
mae = 0
for (x, y_, y_p) in zip(X.values, y.values, y_pred):
    print(f'Actual: {y_}', end=' ')
    print(f'Predicted: {y_p}', end=' ')
    print(f'E: {(y_ - y_p)}')
    me += (y_ - y_p)
    mae += abs(y_ - y_p)
    mse += (y_ - y_p)**2

print(f'ME: {me/len(y)}')
print(f'MAE: {mae/len(y)}')
print(f'MSE: {mse/len(y)}')
