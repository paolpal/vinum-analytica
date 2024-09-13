from vinum_analytica.data import WineDatasetManager # type: ignore
from vinum_analytica.models import RegressorNeuralNetworkModel # type: ignore

from sklearn.metrics import mean_squared_error

print('Loading dataset...', end='')
dataset = WineDatasetManager(y_label='price')
dataset.load('./data/processed/test_regression.csv')
print('done')

model = RegressorNeuralNetworkModel.load('./models/regressor_nn_model.pkl')

# Valuta il modello
print('Evaluating model...', end='')
mse = model.evaluate(dataset, mean_squared_error)
print('done')

print(f'Mean Squared Error: {mse}')
print()

X, y = dataset.get_x(), dataset.get_y()
# Esegui una predizione
print('Predicting...', end='')
y_pred = model.predict(X)
print('done')

mse = 0
me = 0
mae = 0
for (x, y_, y_p) in zip(X.values, y.values, y_pred):
    me += (y_ - y_p)
    mae += abs(y_ - y_p)
    mse += (y_ - y_p)**2

print('Regressor Neural Network Model:')
print(f'ME: {me/len(y)}')
print(f'MAE: {mae/len(y)}')
print(f'MSE: {mse/len(y)}')