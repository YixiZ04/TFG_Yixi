## Import MNIST dataset


from sklearn.datasets import fetch_openml
mnist = fetch_openml ('mnist_784', version = 1, as_frame = False)
mnist.keys ()

## El label está guardado bajo el key "target", y los datos bajo el key de "data"

X, y = mnist ["data"], mnist ["target"]

X.shape

y.shape

## Para ver una imagen, que es de 28x28 píxeles con intensidad de 0-255

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape (28, 28)

plt.imshow (some_digit_image, cmap = "binary")
plt.axis ("off")
plt.show ()

print (y[0])

#Vemos la imagen del número escrito a mano. Parece ser 5, y su label es 5

## Convertimos el label en int en vez de str

import numpy as np

y = y.astype (np.uint8)

## Vamos a crear un test set y un train set

X_train = X [ :60000]
y_train = y [ :60000]
X_test = X [ 60000:]
y_test = y [ 60000:]


## Clasificación binaria: solo identificar 5 y dígitos NO 5.

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier (random_state = 42)
sgd_clf.fit (X_train, y_train_5)

sgd_clf.predict ([some_digit]) #Una predicción particular

## Evaluar clasificadores. Cross-validation.

from sklearn.model_selection import cross_val_score

cross_val_score (sgd_clf, X_train, y_train_5, cv = 3, scoring = "accuracy")

## Vamos a crear un nuevo estimador para los números que no sean 5.

from sklearn.base import BaseEstimator

class Never5Classifier (BaseEstimator):
    """
    Lo que devuelve siempre va a ser un booleano "false" para la matriz de entrada.
    """
    def fit (self, X, y = None):
        pass
    def predict (self, X):
        return np.zeros ((len(X),1), dtype =bool)

never_5_clf = Never5Classifier ()

cross_val_score (never_5_clf, X_train, y_train_5, cv = 3, scoring = "accuracy")

# Devuelve una precisión de 90%. Esto es porque el 90% de las imagenes no son 5.

## Matriz de confunsión (Confusion Matrix).

# Require una predicción previa.

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict (sgd_clf, X_train, y_train_5, cv = 3) #Es simplemente una predicción en los subsets.

print (y_train_pred)

#Para construie la matriz de confusión

from sklearn.metrics import confusion_matrix

confusion_matrix (y_train_5, y_train_pred)

# Vemos la curva de los valores de las predicciones.

y_scores = cross_val_predict (sgd_clf, X_train, y_train_5, cv = 3, method = "decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve (y_train_5, y_scores)

def plot_precision_recall_vs_threshold (precisions, recalls, thresholds):
    plt.plot (thresholds, precisions [:-1], "b--", label = "Precision")
    plt.plot (thresholds, recalls [:-1], "g-", label = "Recall")
    plt.xlim (-50000, 50000)
    plt.legend ()
    plt.xlabel ("Threshold")
    plt.grid ()

plot_precision_recall_vs_threshold (precisions, recalls, thresholds)
plt.show ()

def plot_precision_vs_recall (precisions, recalls):
    plt.plot (recalls, precisions)
    plt.xlabel ("recall")
    plt.ylabel ("precision")
    plt.ylim (0,1)
    plt.xlim (0,1)
    plt.grid ()

plot_precision_vs_recall (precisions, recalls)
plt.show ()

#Suponiendo que queremos una precisión de 90%

from sklearn.metrics import precision_score, recall_score
threshold_90_precision = thresholds [np.argmax (precisions >= 0.90)]
y_train_pred_90 = (y_scores >= threshold_90_precision)
precision_score (y_train_5, y_train_pred_90)
recall_score (y_train_5, y_train_pred_90)

## Curva ROC

from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve (y_train_5, y_scores) #y_scores son los valores de función de predicción

def plot_ROC_curve (fpr, tpr, label = None):
    plt.plot (fpr, tpr, linewidth = 2, label = label)
    plt.plot ([0,1],[0,1], "k--")
    plt.xlabel ("False Positive Rate")
    plt.ylabel ("True Positive Rate (Recall)")
    plt.grid ()
    plt.xlim (0,1)
    plt.ylim (0,1)

plot_ROC_curve (fpr, tpr, label = "ROC Curve")
plt.show ()

#Area under curve (AUC).

from sklearn.metrics import roc_auc_score

roc_auc_score (y_train_5, y_scores)

##Clasificador multiclase

#SVC. OvO

from sklearn.svm import SVC

svm_clf = SVC ()
svm_clf.fit (X_train, y_train) # == Entrenar el modelo
svm_clf.predict ([some_digit])

some_digit_scores = svm_clf.decision_function ([some_digit])
some_digit_scores #10 scores, uno para cada dígito

svm_clf.classes_ #Donde guarda las categorías

#Podemos forzar que use One vs the Rest method.

from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier (SVC())
ovr_clf.fit (X_train, y_train)
ovr_clf.predict ([some_digit])

#SGD (por defecto es OVR)
sgd_clf.fit (X_train, y_train)
sgd_clf.predict ([some_digit])

sgd_clf.decision_function ([some_digit]) #Está mal predicho

## Evaluación de clasificadores multiclase

#Cross-validation

cross_val_score (sgd_clf, X_train, y_train, cv = 3, scoring = "accuracy") #array([0.87365, 0.85835, 0.8689 ])

#Podemos escalar los inputs para mejorar la precisión

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler ()
X_train_scaled = scaler.fit_transform (X_train.astype (np.float64))
cross_val_score (sgd_clf, X_train_scaled, y_train, cv = 3, scoring = "accuracy") #array([0.8983, 0.891 , 0.9018])


## Análisis de error

y_train_pred = cross_val_predict (sgd_clf, X_train_scaled, y_train, cv = 3)
conf_mx = confusion_matrix (y_train, y_train_pred)
conf_mx

plt.matshow (conf_mx, cmap = plt.cm.gray)
plt.show ()

#Normalización de la matriz.

row_sums = conf_mx.sum (axis = 1, keepdims = True)
norm_conf_mx = conf_mx / row_sums

# Hacemos que la diagonal principal sea 0

np.fill_diagonal (norm_conf_mx, 0)
plt.matshow (norm_conf_mx, cmap = plt.cm.gray)
plt.show ()

## Queremos ver algunas imágenes clasificados como 3 y 5, y la confusión.

cl_a, cl_b = 3, 5

X_aa = X_train [(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train [(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train [(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train [(y_train == cl_b) & (y_train_pred == cl_b)]

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

plt.figure (figsize = (8,8))
plt.subplot (221); plot_digits (X_aa [:25], images_per_row = 5)
plt.subplot (222); plot_digits (X_ab [:25], images_per_row = 5)
plt.subplot (223); plot_digits (X_ba [:25], images_per_row = 5)
plt.subplot (224); plot_digits (X_bb [:25], images_per_row = 5)
plt.show ()

## Clasificación multietiqueta.

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7) #Devuelve un booleano para cada celda.
y_train_odd = (y_train % 2 == 1) #Igual
y_multilabel = np.c_[y_train_large, y_train_odd] #Junta los booleanos en una lista para cada casilla

knn_clf = KNeighborsClassifier ()
knn_clf.fit (X_train, y_multilabel)

knn_clf.predict ([some_digit]) #Devuelve el resultado que buscábamos. NO es un número grande y es impar

## Validación de clasificadores multiclase.

y_train_knn_pred = cross_val_predict (knn_clf, X_train, y_multilabel, cv = 3)
from sklearn.metrics import f1_score
f1_score (y_multilabel, y_train_knn_pred, average = "macro") #Supone que cada etiqueta es importante.

## Clasificación multioutput.

noise = np.random.randint (0, 100, (len (X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint (0, 100, (len (X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

plt.figure (figsize = (8,8))
plt.subplot (121); plot_digit (X_test_mod [0])
plt.subplot (122); plot_digit (y_test_mod [0])
plt.show ()

knn_clf.fit (X_train_mod, y_train_mod)
clean_digit = knn_clf.predict ([X_test_mod[0]])
plot_digit (clean_digit)
plt.show ()