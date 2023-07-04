from abc import ABC, abstractmethod

class IWrapper(ABC):
    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...


class BasicWrapper(IWrapper):
    def __init__(self, model, bool_function) -> None:
        self.model = model
        self.predicate = bool_function
        
    def fit(self, X, y):
        xy_data = zip(X, y)
        X_, y_ = zip(*filter(self.predicate, xy_data))

        (self.model).fit(X_, y_)

        return self

    def predict(self, X):
        return self.model.predict(X)
    

from sklearn.decomposition import PCA

class PCAWrapper(IWrapper):
    def __init__(self, model, bool_function) -> None:
        self.model = model
        self.predicate = bool_function

    def fit(self, X, y):
        self.pca = PCA(n_components=128)
        tmp_x = self.pca.fit_transform(X)

        xy_data = zip(tmp_x, y)
        X_, y_ = zip(*filter(self.predicate, xy_data))

        self.model.fit(X_, y_)

        return self
    
    def predict(self, X):
        tmp = self.pca.transform(X)

        return self.model.predict(tmp)
    

"""
class KerasWrapper(IWrapper):
    def __init__(self, model: Sequential, bool_function) -> None:
        self.model = model
        self.predicate = bool_function

    def fit(self, X, y):
        xy_data = zip(X, y)
        X_, y_ = zip(*filter(self.predicate, xy_data))
        
        # To figure out that this retype is necessary only took 3 hours of debugging
        X_ = np.array(X_)
        y_ = np.array(y_)

        tmp_y = to_categorical(y_, num_classes=NUM_CLASSES)

        (self.model).fit(X_, tmp_y, epochs=10)

        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=-1)
    

class TransferKerasWrapper(IWrapper):
    def __init__(self, model: Sequential, bool_function, shape) -> None:
        self.model = model
        self.predicate = bool_function
        self.shape = shape

    def fit(self, X, y):
        xy_data = zip(X, y)
        X_, y_ = zip(*filter(self.predicate, xy_data))
        
        # To figure out that this retype is necessary only took 3 hours of debugging
        X_ = np.array(X_)#.reshape((len(y_),*self.shape))
        y_ = np.array(y_)

        tmp_y = to_categorical(y_, num_classes=NUM_CLASSES)

        (self.model).fit(X_, tmp_y, epochs=1)

        return self

    def predict(self, X):
        tmp = np.array(X)#.reshape((len(X),*self.shape))
        return np.argmax(self.model.predict(tmp), axis=-1)
"""