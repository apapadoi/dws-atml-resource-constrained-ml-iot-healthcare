# Based on the implementaion of Eustache Diemert <eustache@diemert.fr> in scikit-learn package

import gc
import time
from collections import defaultdict
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sklearn as sk
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score

sns.set_style("whitegrid")

from uci_har_utils import read_dataset

RANDOM_STATE = 42

X_train, y_train, X_val, y_val, X_test, y_test = read_dataset()

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
sk.utils.check_random_state(RANDOM_STATE)


def atomic_benchmark_estimator(estimator, X_test, verbose=False):
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_instances, dtype=float)
    for i in range(n_instances):
        instance = X_test[[i], :]
        start = time.time()
        estimator.predict(instance)
        runtimes[i] = time.time() - start
    if verbose:
        print(
            "atomic_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return runtimes


def bulk_benchmark_estimator(estimator, X_test, n_bulk_repeats, verbose):
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_bulk_repeats, dtype=float)
    for i in range(n_bulk_repeats):
        start = time.time()
        estimator.predict(X_test)
        runtimes[i] = time.time() - start
    runtimes = np.array(list(map(lambda x: x / float(n_instances), runtimes)))
    if verbose:
        print(
            "bulk_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return runtimes


def mini_bulk_benchmark_estimator(estimator, X_test, batch_divisor, verbose):
    n_instances = X_test.shape[0]
    batch_size = n_instances // batch_divisor
    runtimes = []
    for start_index in range(0, len(X_test), batch_size):
        end_index = start_index + batch_size
        batch_df = X_test[start_index:end_index]
        start_time = time.time()
        estimator.predict(batch_df)
        runtimes.append((time.time() - start_time) / batch_df.shape[0])

    runtimes = np.array(runtimes, dtype=float)

    if verbose:
        print(
            "bulk_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return runtimes

def benchmark_estimator(estimator, X_test, n_bulk_repeats=30, verbose=False):
    atomic_runtimes = atomic_benchmark_estimator(estimator, X_test, verbose)
    bulk_runtimes = bulk_benchmark_estimator(estimator, X_test, n_bulk_repeats, verbose)
    mini_bulk_runtimes = mini_bulk_benchmark_estimator(estimator, X_test, 10, verbose)
    return atomic_runtimes, bulk_runtimes, mini_bulk_runtimes


def barplot_metrics(stats, configuration):
    names = [model['name'] for model in configuration["estimators"]]
    accuracies = [stats[model]['accuracy'] for model in names]
    macro_f1_scores = [stats[model]['macro_f1_score'] for model in names]

    bar_width = 0.35

    r1 = np.arange(len(names))
    r2 = [x + bar_width for x in r1]

    fig, ax1 =  plt.subplots(figsize=(10, 6))
    bars1 = plt.bar(r1, accuracies, width=bar_width, edgecolor='grey', label='Accuracy')
    bars2 = plt.bar(r2, macro_f1_scores, width=bar_width, edgecolor='grey', label='Macro F1 Score')

    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.xticks([r + bar_width / 2 for r in range(len(names))], names)
    plt.ylim(0, 1)  # Assuming the metrics are in the range [0, 1]
    plt.title('Performance of Models')
    
    cls_infos = [
        "%s\n(%d %s)"
        % (
            estimator_conf["name"],
            estimator_conf["complexity_computer"](estimator_conf["instance"]),
            estimator_conf["complexity_label"],
        )
        for estimator_conf in configuration["estimators"]
    ]
    plt.setp(ax1, xticklabels=cls_infos)

    plt.legend()

    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')
    
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')


    plt.show()


def boxplot_runtimes(runtimes, pred_type, configuration):
    fig, ax1 = plt.subplots(figsize=(15, 15))
    bp = plt.boxplot(
        runtimes,
    )

    medians = [item.get_ydata()[0] for item in bp['medians']]

    cls_infos = [
        "%s\n(%d %s) Median=%0.3f"
        % (
            estimator_conf["name"],
            estimator_conf["complexity_computer"](estimator_conf["instance"]),
            estimator_conf["complexity_label"],
            median
        )
        for estimator_conf, median in zip(configuration["estimators"], medians)
    ]
    plt.setp(ax1, xticklabels=cls_infos)
    plt.setp(bp["boxes"], color="black")
    plt.setp(bp["whiskers"], color="black")
    plt.setp(bp["fliers"], color="red", marker="+")

    ax1.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

    ax1.set_axisbelow(True)
    ax1.set_title(
        "Prediction Time per Instance - %s, %d feats WITHOUT input validation checks."
        % (pred_type.capitalize(), configuration["n_features"])
    )

    ax1.set_ylabel("Prediction Time (us)")

    plt.show()


def benchmark(configuration):
    stats = {}
    for estimator_conf in configuration["estimators"]:
        print("Benchmarking", estimator_conf["instance"])
        estimator_conf["instance"].fit(X_train, y_train)
        gc.collect()
        a, b, c = benchmark_estimator(estimator_conf["instance"], X_test)
        stats[estimator_conf["name"]] = {"atomic": a, "bulk": b, "mini-bulk": c}

        y_pred = estimator_conf["instance"].predict(X_test)
        stats[estimator_conf["name"]]["accuracy"] = accuracy_score(y_test, y_pred)
        stats[estimator_conf["name"]]["macro_f1_score"] = f1_score(y_test, y_pred, average='macro')
        

    cls_names = [
        estimator_conf["name"] for estimator_conf in configuration["estimators"]
    ]
    runtimes = [1e6 * stats[clf_name]["atomic"] for clf_name in cls_names]
    boxplot_runtimes(runtimes, "atomic", configuration)

    runtimes = [1e6 * stats[clf_name]["bulk"] for clf_name in cls_names]
    boxplot_runtimes(runtimes, "bulk (%d)" % configuration["n_test"], configuration)

    runtimes = [1e6 * stats[clf_name]["mini-bulk"] for clf_name in cls_names]
    boxplot_runtimes(runtimes, "mini-bulk (%d)" % configuration["n_test"], configuration)

    barplot_metrics(stats, configuration)


configuration = {
    "n_train": X_train.shape[0],
    "n_test": X_test.shape[0],
    "n_features":X_train.shape[1],
    "estimators": [
        {
            "name": "SGDClassifier",
            "instance": SGDClassifier(random_state=RANDOM_STATE),
            "complexity_label": "non-zero coefficients",
            "complexity_computer": lambda clf: np.count_nonzero(clf.coef_),
        },
        {
            "name": "RandomForest",
            "instance": RandomForestClassifier(random_state=RANDOM_STATE),
            "complexity_label": "estimators",
            "complexity_computer": lambda clf: clf.n_estimators,
        },
        {
            "name": "SVC with rbf",
            "instance": SVC(random_state=RANDOM_STATE),
            "complexity_label": "support vectors",
            "complexity_computer": lambda clf: len(clf.n_support_),
        },
    ],
}

with sk.config_context(assume_finite=True): # uncomment if you want to allow sklearn to do validation
    benchmark(configuration)

# NOTE: cant benefit from sparse formats
# def sparsity_ratio(X_test):
#     return 1.0 - np.count_nonzero(X_test) / float(X_test.shape[0] * X_test.shape[1])
# print("input sparsity ratio:", sparsity_ratio(X_test))