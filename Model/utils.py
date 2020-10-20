import re
import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = str(string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def focal_loss(y_true, y_pred, class_weight=2, gamma=2.):
    # Took from: https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    class_weight_tf = tf.constant(class_weight, dtype=tf.float32)

    epsilon = 1.e-9

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(class_weight_tf, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    loss = tf.reduce_mean(reduced_fl, axis=-1)
    return loss


def loss_func(class_weight):
    def loss(y_true, y_pred):
        return focal_loss(y_true, y_pred, class_weight)

    return loss

def flatten_list(l):
    return [item for sublist in l for item in sublist]


def compute_class_weight(class_count):
    y = [[i] * v for i, v in enumerate(class_count)]
    y = flatten_list(y)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
    return class_weights
