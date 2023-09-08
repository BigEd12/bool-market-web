import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
def preprocess(X, y):

    def onehotencode(y_synth):
        y_synth = np.array(y_synth)
        start = y_synth[:,0]
        end = y_synth[:,1]
        pattern = y_synth[:,2]
        mapping = {
            0: [0, 0, 0, 0, 1],
            1: [0, 0, 0, 1, 0],
            2: [0, 0, 1, 0, 0],
            3: [0, 1, 0, 0, 0],
            4: [1, 0, 0, 0, 0]
        }
        mapped_values = [list(mapping[val]) for val in pattern]
        y_s = (np.column_stack((start, end, mapped_values)))
        return y_s

    y_preprocessed = onehotencode(y)

    y_dates = y_preprocessed[:,:2]
    y_p = y_preprocessed[:,2:]

    X_train_processed=[]
    scale={}
    for i in range(len(X)):
        open_scaler_i = StandardScaler()
        X_train_processed.append(open_scaler_i.fit_transform(X[i]))
        scale[f"open_scaler_{len(X[i])}"] = open_scaler_i

    maxlen = 450

    X_train_preprocessed = pad_sequences(X_train_processed, maxlen = maxlen, dtype='float32', padding='post', value=-1)

    X_train_preprocessed = tf.convert_to_tensor(X_train_preprocessed, np.float32)
    y_train_p = tf.convert_to_tensor(y_p, np.int16)
    y_train_dates = tf.convert_to_tensor(y_dates, np.int16)

    return X_train_preprocessed, y_train_p, y_train_dates


def preprocess_X(X):

    assert len(X)<451, "The maximum days possible are 450"



    open_scaler = StandardScaler()
    X_train_processed = open_scaler.fit_transform(X)


    maxlen = 450

    X_train_preprocessed = pad_sequences([X_train_processed], maxlen = maxlen, dtype='float32', padding='post', value=-1)

    if len(X_train_preprocessed.shape) == 2:
        X_train_preprocessed = tf.convert_to_tensor([X_train_preprocessed], np.float32)
    elif len(X_train_preprocessed.shape) == 3:
        X_train_preprocessed = tf.convert_to_tensor(X_train_preprocessed, np.float32)
    else:
        return "Errorrrrrr"

    return X_train_preprocessed, open_scaler


def hello_world():
    return "Hello World"