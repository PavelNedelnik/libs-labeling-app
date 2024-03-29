from segmentation.wrappers import BasicWrapper, PCAWrapper
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

filtering_predicate = lambda x: x[1] >= 0

models = [
    KMeans(n_clusters=3, n_init='auto'),
    BasicWrapper(KNeighborsClassifier(n_jobs=-1), filtering_predicate),
    BasicWrapper(RandomForestClassifier(max_depth=3), filtering_predicate),
    PCAWrapper(MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=300, activation='relu', solver='adam', random_state=1), filtering_predicate),
]

# maps model names to indices over <models> array
model_names = list(enumerate([
        'Naive KMeans', 'KNN', 'Random Forest', 'PCA_MLP'
]))
