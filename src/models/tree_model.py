from sklearn.tree import DecisionTreeClassifier
from .model import Model
import pickle

class TreeModel(Model):
    def __init__(self, vectorizer = None, criterion='gini', min_impurity_decrease=0.0, max_depth=None):
        self.model = DecisionTreeClassifier(criterion=criterion, min_impurity_decrease=min_impurity_decrease, max_depth=max_depth)
        self.model.set_params(class_weight='balanced')
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
        self.model = DecisionTreeClassifier(max_depth=self.model.max_depth)
        self.model.set_params(class_weight='balanced')
        self.vectorizer = None

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump((self.model, self.vectorizer, self.hyperparameters), file)

    @classmethod
    def load(cls, path) -> 'TreeModel':
        with open(path, 'rb') as file:
            model, vectorizer, hyperparameters = pickle.load(file)
            tree = cls(vectorizer=vectorizer, **hyperparameters)
            tree.model = model
        return tree

    def classes(self):
        return self.model.classes_
    