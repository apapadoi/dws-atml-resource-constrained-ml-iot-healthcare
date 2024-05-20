import pathlib
import random

import sklearn as sk
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from tensorflow_model_optimization.python.core.keras.compat import keras

from sklearn.metrics import accuracy_score, f1_score

from uci_har_utils import read_dataset, evaluate_model


RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 3
NUM_EPOCHS = 1000
BATCH_SIZE = 64
MODEL_OUTPUT_FOLDER = "../models/uci_har/quantization_aware_training"
DEVICE = "/device:GPU:1"
FINE_TUNE_SIZE = 1000

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
sk.utils.check_random_state(RANDOM_STATE)

X_train, y_train, X_val, y_val, X_test, y_test = read_dataset()

print('Training')
print(X_train.shape)
print(len(y_train))
print(y_train.value_counts())

print('Validation')
print(X_val.shape)
print(len(y_val))
print(y_val.value_counts())

print('Testing')
print(X_test.shape)
print(len(y_test))
print(y_test.value_counts())

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Was built with GPU support: ", tf.test.is_built_with_cuda())

with tf.device(DEVICE):
    y_train_encoded = to_categorical(np.array(y_train.tolist()))
    y_test_encoded = to_categorical(np.array(y_test.tolist()))
    y_val_encoded = to_categorical(np.array(y_val.tolist()))

    model = Sequential()
    model.add(Dense(units=y_train.nunique(), activation='softmax', input_dim=X_train.shape[1]))

    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=[
            'accuracy', 
            tf.keras.metrics.F1Score(average='macro')
        ]
    )

    model.fit(
        X_train, 
        y_train_encoded, 
        epochs=NUM_EPOCHS, 
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val_encoded),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE)
        ]
    )

    test_results = model.evaluate(X_test, y_test_encoded)

    print(test_results)

    quantize_model = tfmot.quantization.keras.quantize_model

    quantization_aware_model = quantize_model(model)

    quantization_aware_model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=[
            'accuracy', 
            tf.keras.metrics.F1Score(average='macro')
        ]
    )

    quantization_aware_model.summary()

    # fine-tune with quantization aware training on a subset of the training data for 1 epoch
    training_subset_indices = np.random.randint(0, X_train.shape[0], size=FINE_TUNE_SIZE)
    X_train_subset = X_train[training_subset_indices]
    y_train_subset = y_train.iloc[training_subset_indices]

    y_train_subset_encoded = to_categorical(np.array(y_train_subset.tolist()))

    quantization_aware_model.fit(
        X_train_subset, 
        y_train_subset_encoded, 
        epochs=1, 
        batch_size=BATCH_SIZE,
        validation_split=0.1
    )

    baseline_model_results = model.evaluate(X_test, y_test_encoded, verbose=0)

    quantization_aware_model_results = quantization_aware_model.evaluate(X_test, y_test_encoded, verbose=0)

    print('Baseline test results on TensorFlow backend:', baseline_model_results)
    print('Quantization aware test results on TensorFlow backend:', quantization_aware_model_results)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_models_dir = pathlib.Path(MODEL_OUTPUT_FOLDER)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    tflite_model_file = tflite_models_dir/"initial_model.tflite"
    tflite_model_file.write_bytes(tflite_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(quantization_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantization_aware_tflite_model = converter.convert()

    quantization_aware_file = tflite_models_dir/"model_quant.tflite"
    quantization_aware_file.write_bytes(quantization_aware_tflite_model)
    
    y_pred_model_quant = evaluate_model(quantization_aware_file, X_test)

print("Quantization aware test accuracy on TFLite backend: ", accuracy_score(y_test, y_pred_model_quant))
print("Quantization aware test macro F1-score on TFLite backend: ", f1_score(y_test, y_pred_model_quant, average='macro'))