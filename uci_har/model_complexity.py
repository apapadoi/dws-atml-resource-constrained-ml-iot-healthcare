# Authors: Eustache Diemert <eustache@diemert.fr>
#          Maria Telenczuk <https://github.com/maikia>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: BSD 3 clause

import time
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sklearn as sk
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import hamming_loss, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC

sns.set_style("whitegrid")

from uci_har_utils import read_dataset


RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
sk.utils.check_random_state(RANDOM_STATE)


X_train, y_train, X_val, y_val, X_test, y_test = read_dataset()

def benchmark_influence(conf):
    """
    Benchmark influence of `changing_param` on both MSE and latency.
    """
    prediction_times = []
    prediction_powers = []
    complexities = []
    for param_value in conf["changing_param_values"]:
        conf["tuned_params"][conf["changing_param"]] = param_value
        estimator = conf["estimator"](**conf["tuned_params"])

        print("Benchmarking %s" % estimator)
        estimator.fit(conf["data"]["X_train"], conf["data"]["y_train"])
        conf["postfit_hook"](estimator)
        complexity = conf["complexity_computer"](estimator)
        complexities.append(complexity)
        start_time = time.time()
        for _ in range(conf["n_samples"]):
            y_pred = estimator.predict(conf["data"]["X_test"])
        elapsed_time = (time.time() - start_time) / float(conf["n_samples"])
        prediction_times.append(elapsed_time * 1000)
        pred_score = conf["prediction_performance_computer"](
            conf["data"]["y_test"], y_pred
        )
        prediction_powers.append(pred_score)
        print(
            "Complexity: %d | %s: %.4f | Pred. Time: %fs\n"
            % (
                complexity,
                conf["prediction_performance_label"],
                pred_score,
                elapsed_time,
            )
        )
    
    combined = list(zip(prediction_powers, prediction_times, complexities))
    sorted_combined = sorted(combined, key=lambda x: x[2])
    prediction_powers_sorted, prediction_times_sorted, complexities_sorted = zip(*sorted_combined)

    return prediction_powers_sorted, prediction_times_sorted, complexities_sorted


def _count_nonzero_coefficients(estimator):
    a = estimator.coef_
    return np.count_nonzero(a)


configurations = [
    # {
    #     "estimator": SGDClassifier,
    #     "tuned_params": {
    #         "penalty": "elasticnet",
    #         "random_state": RANDOM_STATE
    #     },
    #     "changing_param": "l1_ratio",
    #     "changing_param_values": np.arange(0.01, 1.0, 0.01).tolist(),
    #     "complexity_label": "non_zero coefficients",
    #     "complexity_computer": _count_nonzero_coefficients,
    #     "prediction_performance_computer": lambda x, y: f1_score(x, y, average='macro'),
    #     "prediction_performance_label": "Macro f1-score",
    #     "postfit_hook": lambda x: x,
    #     "data": {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
    #     "n_samples": 10,
    # },
    # {
    #     "estimator": RandomForestClassifier,
    #     "tuned_params": {
    #         "random_state": RANDOM_STATE
    #     },
    #     'changing_param': 'n_estimators',
    #     'changing_param_values': [10, 25, 50, 75, 100, 125, 150, 175, 200],
    #     'complexity_label': 'n_trees',
    #     'complexity_computer': lambda x: x.n_estimators,
    #     'data': {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
    #     'postfit_hook': lambda x: x,
    #     'prediction_performance_computer': lambda x, y: f1_score(x, y, average='macro'),
    #     'prediction_performance_label': 'Macro f1-score',
    #     'n_samples': 10,
    # },
    {
        'estimator': NuSVC,
        'tuned_params': {
            "random_state": RANDOM_STATE
        },
        'changing_param': 'nu',
        'changing_param_values': np.array([0.01, 0.02, 0.03, 0.04]).tolist() + np.arange(0.05, 0.85, 0.05).tolist(),
        'complexity_label': 'n_support_vectors',
        'complexity_computer': lambda x: len(x.support_vectors_),
        'data': {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        'postfit_hook': lambda x: x,
        'prediction_performance_computer': lambda x, y: f1_score(x, y, average='macro'),
        'prediction_performance_label': 'Macro f1-score',
        'n_samples': 10
    },
]

def plot_influence(conf, mse_values, prediction_times, complexities):
    """
    Plot influence of model complexity on both accuracy and latency.
    """

    fig = plt.figure()
    fig.subplots_adjust(right=0.75)

    # first axes (prediction error)
    ax1 = fig.add_subplot(111)
    line1 = ax1.plot(complexities, mse_values, c="tab:blue", ls="-")[0]
    ax1.set_xlabel("Model Complexity (%s)" % conf["complexity_label"])
    y1_label = conf["prediction_performance_label"]
    ax1.set_ylabel(y1_label)

    ax1.spines["left"].set_color(line1.get_color())
    ax1.yaxis.label.set_color(line1.get_color())
    ax1.tick_params(axis="y", colors=line1.get_color())

    # second axes (latency)
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    line2 = ax2.plot(complexities, prediction_times, c="tab:orange", ls="-")[0]
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    y2_label = "Time (ms)"
    ax2.set_ylabel(y2_label)
    ax1.spines["right"].set_color(line2.get_color())
    ax2.yaxis.label.set_color(line2.get_color())
    ax2.tick_params(axis="y", colors=line2.get_color())

    plt.legend(
        (line1, line2), ("prediction macro f1-score", "prediction latency"), loc="best"
    )

    plt.title(
        "Influence of varying '%s' on %s"
        % (conf["changing_param"], conf["estimator"].__name__)
    )


for conf in configurations:
    prediction_performances, prediction_times, complexities = benchmark_influence(conf)
    plot_influence(conf, prediction_performances, prediction_times, complexities)
plt.show()