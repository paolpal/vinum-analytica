from sklearn.tree import DecisionTreeRegressor
from .model import Model
import pickle

class RegressorTreeModel(Model):
    def __init__(self, vectorizer=None, criterion='friedman_mse', min_impurity_decrease=0.0, max_depth=None):
        self.model = DecisionTreeRegressor(criterion=criterion, min_impurity_decrease=min_impurity_decrease, max_depth=max_depth)
        self.vectorizer = vectorizer
        self.hyperparameters = {
            'criterion': criterion,
            'min_impurity_decrease': min_impurity_decrease,
            'max_depth': max_depth
        }

    def train(self, train, **kwargs):
        self.model.fit(train.get_x(), train.get_y())

    def predict(self, x, **kwargs):
        token = self.vectorizer.transform(x)
        return self.model.predict(token)
    
    def reset(self):
        self.model = DecisionTreeRegressor(max_depth=self.model.max_depth)
        self.vectorizer = None

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump((self.model, self.vectorizer, self.hyperparameters), file)

    @classmethod
    def load(cls, path) -> 'RegressorTreeModel':
        with open(path, 'rb') as file:
            model, vectorizer, hyperparameters = pickle.load(file)
            tree = cls(vectorizer=vectorizer, **hyperparameters)
            tree.model = model
        return tree

    def feature_importances(self):
        return self.model.feature_importances_
    
