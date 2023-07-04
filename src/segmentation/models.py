from segmentation.wrappers import BasicWrapper, PCAWrapper  #, KerasWrapper, TransferKerasWrapper
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

# from segmentation.cnn_model import cnn_model
# from segmentation.keras_model import keras_model
# from segmentation.transfer_model import transfer_model

filtering_predicate = lambda x: x[1] >= 0

models = [KMeans(n_clusters=3, n_init='auto'),
          #BasicWrapper(GaussianNB(), filtering_predicate),
          BasicWrapper(KNeighborsClassifier(n_jobs=-1), filtering_predicate),
          BasicWrapper(RandomForestClassifier(max_depth=3), filtering_predicate),
          #BasicWrapper(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), filtering_predicate),
          #BasicWrapper(MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam', random_state=1), filtering_predicate),
          #BasicWrapper(MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=300, activation='relu', solver='adam', random_state=1), filtering_predicate),
          #PCAWrapper(SVC(random_state=0), filtering_predicate),
          PCAWrapper(MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=300, activation='relu', solver='adam', random_state=1), filtering_predicate),
          #KerasWrapper(keras_model, filtering_predicate),
          #TransferKerasWrapper(transfer_model, filtering_predicate, (88,44,1)),
          #KerasWrapper(cnn_model, filtering_predicate),
          ]

# maps model names to indices over <models> array
model_names = {name: i for i, name in enumerate(
    ['Naive KMeans', 'Bayes', 'KNN', 'Random Forest', 'Gradient Boosting', 'MLP', 'MLP_bigger', 'PCA_SVM', 'PCA_MLP', 'Keras_NN', "Transfer_model", "1D_CNN"])}
