import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split


def read_dataset():
    X_TRAIN_INPUT_FILE = "../datasets/UCI_HAR_Dataset/train/X_train.txt"
    X_TEST_INPUT_FILE = "../datasets/UCI_HAR_Dataset/test/X_test.txt"

    Y_TRAIN_INPUT_FILE = "../datasets/UCI_HAR_Dataset/train/y_train.txt"
    Y_TEST_INPUT_FILE = "../datasets/UCI_HAR_Dataset/test/y_test.txt"

    def read_uci_har_txt(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        data = []

        for line in lines:
            numbers = [float(num) for num in line.split()]
            data.append(numbers)

        df = pd.DataFrame(data)
        return df


    def map_classes(initial_value):
        return initial_value - 1


    X_train = read_uci_har_txt(X_TRAIN_INPUT_FILE)
    y_train = pd.read_csv(Y_TRAIN_INPUT_FILE, header=None)[0]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/7, stratify=y_train) # take 10% of the initial dataset as validation

    X_train = X_train.values.reshape(-1, X_train.shape[1])
    y_train = y_train.apply(map_classes)

    X_val = X_val.values.reshape(-1, X_val.shape[1])
    y_val = y_val.apply(map_classes)

    X_test = read_uci_har_txt(X_TEST_INPUT_FILE)
    y_test = pd.read_csv(Y_TEST_INPUT_FILE, header=None)[0]

    X_test = X_test.values.reshape(-1, X_test.shape[1])
    y_test = y_test.apply(map_classes)

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(model_path, X_test, int_quant=False):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    for test_instance in X_test:
        if int_quant and input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            test_instance = test_instance / input_scale + input_zero_point

        test_instance = np.expand_dims(test_instance, axis=0).astype(np.float32 if not int_quant else input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], test_instance)

        interpreter.invoke()
        
        output = interpreter.tensor(output_details[0]['index'])
        
        predicted_class = np.argmax(output()[0])
        predictions.append(predicted_class)


    return predictions
