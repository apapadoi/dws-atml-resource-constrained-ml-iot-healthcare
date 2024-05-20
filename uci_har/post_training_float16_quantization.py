import pathlib
import random

import sklearn as sk
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from sklearn.metrics import accuracy_score, f1_score

from uci_har_utils import read_dataset, evaluate_model


RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 3
NUM_EPOCHS = 1000
BATCH_SIZE = 64
MODEL_OUTPUT_FOLDER = "../models/uci_har/post_training_float16_quantization"
DEVICE = "/device:GPU:1"

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

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_models_dir = pathlib.Path(MODEL_OUTPUT_FOLDER)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    tflite_model_file = tflite_models_dir/"initial_model.tflite"
    tflite_model_file.write_bytes(tflite_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_fp16_model = converter.convert()
    tflite_model_fp16_file = tflite_models_dir/"model_quant_f16.tflite"
    tflite_model_fp16_file.write_bytes(tflite_fp16_model)

    y_pred_initial_model = evaluate_model(tflite_model_file, X_test)
    y_pred_model_quant_f16 = evaluate_model(tflite_model_fp16_file, X_test)

print("Initial model accuracy: ", accuracy_score(y_test, y_pred_initial_model))
print("Float-16 quantized model accuracy: ", accuracy_score(y_test, y_pred_model_quant_f16))

print("Initial model macro F1-score: ", f1_score(y_test, y_pred_initial_model, average='macro'))
print("Float-16 quantized model macro F1-score: ", f1_score(y_test, y_pred_model_quant_f16, average='macro'))
