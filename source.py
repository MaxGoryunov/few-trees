from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import time
import openml
from  argparse import ArgumentParser
import sys
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


def print_tree(tree, index=0, offset=0):
    if index == -1:
        return
    print_tree(tree, tree.children_left[index], offset + 1)
    print(' ' * (2 * offset) + f'{offset}:' + str(index))
    print_tree(tree, tree.children_right[index], offset + 1)


# === data functions ===

def dim_spirals(n, d, std_dev=0):
    t = np.linspace(0, 6 * np.pi, n)
    x = []
    for i in range(2, d + 1, 2):
        if std_dev != 0:
            cos_noise = np.random.normal(0, std_dev, n)
            sin_noise = np.random.normal(0, std_dev, n)
            x.append(t * np.cos(t * i / 2) + cos_noise)
            x.append(t * np.sin(t * i / 2) + sin_noise)
        else:
            x.append(t * np.cos(t * i / 2))
            x.append(t * np.sin(t * i / 2))
    x = np.array(x).T
    labels_x = ['A'] * n
    y = []
    for i in range(2, d + 1, 2):
        if std_dev != 0:
            cos_noise = np.random.normal(0, std_dev, n)
            sin_noise = np.random.normal(0, std_dev, n)
            y.append(t * np.cos(t * i / 2 + np.pi) + cos_noise)
            y.append(t * np.sin(t * i / 2 + np.pi) + sin_noise)
        else:
            y.append(t * np.cos(t * i / 2 + np.pi))
            y.append(t * np.sin(t * i / 2 + np.pi))
    y = np.array(y).T
    labels_y = ['B'] * n
    points = np.vstack((x, y))
    labels = labels_x + labels_y
    if d == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:n, 0], points[:n, 1], c='red', label='A')
        plt.scatter(points[n:, 0], points[n:, 1], c='blue', label='B')
        plt.title("Многомерные спирали")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()
    return points, np.array(labels)


def dim_logs_uniforms(d):
    a = []
    for _ in range(d):
        flip = np.random.randint(0, 2)
        if flip % 2 == 0:
            a.append(np.random.uniform(-4, -1))
        else:
            a.append(np.random.uniform(1, 4))
    a = np.array(a)
    return a


def dim_logs(n, d):
    t = np.random.uniform(0.5, 5, n)
    a = dim_logs_uniforms(d)
    print(f'a shape: {a.shape}')
    x = np.array([a_i * np.log(t) for a_i in a]).T
    labels_x = ['A'] * n

    b = dim_logs_uniforms(d)
    y = np.array([b_i * np.log(t) for b_i in b]).T
    labels_y = ['B'] * n

    points = np.vstack((x, y))
    labels = labels_x + labels_y

    if d == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:n, 0], points[:n, 1], c='red', label='A')
        plt.scatter(points[n:, 0], points[n:, 1], c='blue', label='B')
        plt.title("Logarithmic Data Visualization (d=2)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    return points, labels


# === distance calculation ===

def trace_binary_path_opt(tree, sample_leaf_index):
    children_left = tree.children_left
    children_right = tree.children_right
    search = sample_leaf_index
    binary_path = []
    while True:
        i = np.where(children_left == search)[0]
        if i.size:
            binary_path.append(0)
        else:
            i = np.where(children_right == search)[0]
            binary_path.append(1)
        i = i[0]
        if i == 0:
            break
        search = i
    return np.array(binary_path[::-1], dtype=int)


def padded_dist(x: np.ndarray, y: np.ndarray):
    diff_idx = np.where(x != y)[0]
    if diff_idx.size > 0:
        return len(x) - diff_idx[0]
    return 0


def gaussian_dist_flat_bc(points_target, points_other):
    diff = points_target[:, None, :] - points_other[None, :, :]
    return np.sum(diff ** 2, axis=-1)


def scale_flat_dists(flat_dists, pair_dists):
    max_flat = np.max(flat_dists)
    max_pair = np.max(pair_dists)
    if max_flat == 0:
        return np.zeros_like(flat_dists)
    return (flat_dists / max_flat) * max_pair


def get_leaf_indices(tree):
    children_left = tree.children_left
    children_right = tree.children_right
    return np.array([i for i in range(tree.node_count) if children_left[i] == -1 and children_right[i] == -1])


def pad_binary_paths(tree_paths):
    max_length = max(len(path) for path in tree_paths)
    return np.array([np.pad(path, (0, max_length - len(path)), 'constant') for path in tree_paths])


def padded_no_cdist_leaf_dists(tree: ExtraTreeClassifier):
    leaf_indices = get_leaf_indices(tree.tree_)
    padded_paths = pad_binary_paths([trace_binary_path_opt(tree.tree_, lind) for lind in leaf_indices])
    n_paths = len(padded_paths)
    lind_map = {lind: i for i, lind in enumerate(leaf_indices)}
    pair_dists = np.zeros((n_paths, n_paths))
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            pair_dists[i, j] = padded_dist(padded_paths[i], padded_paths[j])
            pair_dists[j, i] = pair_dists[i, j]
    return pair_dists, lind_map


def leaf_precomp_no_cdist_pairwise_distances_with_padded_no_cdist_leaves(extrees: ExtraTreeClassifier, points: np.array, train_idx: np.ndarray):
    n_points = points.shape[0]
    n_trees = len(extrees.estimators_)
    n_train_idx = len(train_idx)
    leaf_indices = np.array([tree.apply(points) for tree in extrees.estimators_])  # Shape: (n_trees, n_points)
    pairwise_distances = np.zeros((n_points, n_train_idx))
    for t in range(n_trees):
        tree = extrees.estimators_[t]
        leaves = leaf_indices[t]
        leaves_train = leaves[train_idx]
        leaf_dists, lind_map = padded_no_cdist_leaf_dists(tree)
        for i in range(n_points):
            for j in range(n_train_idx):
                pairwise_distances[i, j] += leaf_dists[lind_map[leaves[i]], lind_map[leaves_train[j]]]
    pairwise_distances /= n_trees
    return pairwise_distances


def mismatch_metric(extrees: ExtraTreeClassifier, points: np.array, train_idx: np.ndarray):
    n_points = points.shape[0]
    n_train_idx = len(train_idx)
    leaf_indices = np.array([tree.apply(points) for tree in extrees.estimators_]).T  # Shape: (n_points, n_trees)
    pairwise_distances = np.zeros((n_points, n_train_idx))
    for i in range(n_points):
        for j, train_i in enumerate(train_idx):
            pairwise_distances[i, j] += 2 * np.sum(leaf_indices[i] != leaf_indices[train_i])
    return pairwise_distances


def predictions_from_flat_and_pairdist_mix_tree_unsquared(target_indices, train_idx, labels, flat_dists, pair_dists, bandwidth, alpha):
    beta = 1 - alpha
    pairwise_target = pair_dists[target_indices]
    flat_scaled = scale_flat_dists(flat_dists, pairwise_target)
    combined_dists = alpha * (flat_scaled ** 2) + beta * (pairwise_target)
    weights = np.exp(-(combined_dists / (2 * bandwidth ** 2)))
    weights /= np.sum(weights, axis=1, keepdims=True)
    train_labels = labels[train_idx]
    predictions = np.dot(weights, train_labels)
    return predictions


def calculate_acc_auroc_f1(label: str, results: dict[str, dict[str, list[float]]], test_labels: np.ndarray, pred_labels: np.ndarray, pred_probs: np.ndarray) -> None:
    binary_classification = len(np.unique(test_labels)) == 2
    results['Accuracy'][label].append(accuracy_score(test_labels, pred_labels))
    if binary_classification:
        results['AUROC'][label].append(roc_auc_score(test_labels, pred_probs[:, 1]))
        results['F1'][label].append(f1_score(test_labels, pred_labels, average='binary'))
    else:
        results['AUROC'][label].append(roc_auc_score(test_labels, pred_probs, multi_class='ovr', average='macro'))
        results['F1'][label].append(f1_score(test_labels, pred_labels, average='macro'))


def kernel_mix_transform_labels(labels, train_idx):
    if len(labels.shape) == 1:
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoder.fit(labels[train_idx].reshape(-1, 1))
        labels_one_hot = one_hot_encoder.transform(labels.reshape(-1, 1))
    else:
        labels_one_hot = labels
    return labels_one_hot


def get_max_splits(base_labels):
    bl = np.array(base_labels)
    classes, counts = np.unique(bl, return_counts=True)
    return np.min(counts)


def fitted_extrees(
    train_points: np.ndarray, train_labels: np.ndarray, n_estimators: int,
    random_state: int = 42, max_depth: int | None = None
):
    extrees = ExtraTreesClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1)
    extrees.fit(train_points, train_labels)
    return extrees


# === comparison of methods ===

def cross_validate_real_data(
    data_gen, data_prep, n_estimators=10, num_neighbors=1, bandwidth=1,
    alpha=0.5, n_splits=5, n_repeats=2, test_swap=False
):
    flat_dists_calc = gaussian_dist_flat_bc
    mix_dists_calc = predictions_from_flat_and_pairdist_mix_tree_unsquared
    base_points, base_labels = data_gen()
    print(f'max splits = {get_max_splits(base_labels)}')
    print(f'Unique labels: {np.unique(base_labels)}')
    print(f'len(base_points) = {len(base_points)}')
    random_state = 42
    results = {
        'Accuracy': {
            'Metric': [], 'Mismatch': [],  'Mix0.00': [],  'Mix0.25': [],
            'Mix0.50': [], 'Mix0.75': [], 'KR': [], 'ExtraTrees': [], 'KNN': []
        },
        'AUROC': {
            'Metric': [], 'Mismatch': [],  'Mix0.00': [],  'Mix0.25': [],
            'Mix0.50': [], 'Mix0.75': [], 'KR': [], 'ExtraTrees': [], 'KNN': []
        },
        'F1': {
            'Metric': [], 'Mismatch': [],  'Mix0.00': [],  'Mix0.25': [],
            'Mix0.50': [], 'Mix0.75': [], 'KR': [], 'ExtraTrees': [], 'KNN': []
        },
    }
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(base_points, base_labels)):
        if test_swap:
            train_idx, test_idx = test_idx, train_idx
        print(f'Fold = {fold_idx + 1}')
        points, labels = data_prep(base_points, base_labels, train_idx)
        train_points, train_labels = points[train_idx], labels[train_idx]
        test_points, test_labels = points[test_idx], labels[test_idx]
        extrees = fitted_extrees(train_points, train_labels, n_estimators, random_state)
        curt = time.time_ns()
        pair_dists = leaf_precomp_no_cdist_pairwise_distances_with_padded_no_cdist_leaves(extrees, points, train_idx)
        print(f'Pairdist time = {(time.time_ns() - curt) / 1_000_000:.4f}ms')
        train_distances = pair_dists[train_idx]
        test_distances = pair_dists[test_idx]

        knn_precomputed = KNeighborsClassifier(n_neighbors=num_neighbors, metric='precomputed')
        knn_precomputed.fit(train_distances, train_labels)
        predicted_labels = knn_precomputed.predict(test_distances)
        y_pred_proba = knn_precomputed.predict_proba(test_distances)
        calculate_acc_auroc_f1('Metric', results, test_labels, predicted_labels, y_pred_proba)

        knn_mismatch = KNeighborsClassifier(n_neighbors=num_neighbors, metric='precomputed')
        mismatch_dists = mismatch_metric(extrees, points, train_idx)
        knn_mismatch.fit(mismatch_dists[train_idx], train_labels)
        mismatch_pred = knn_mismatch.predict(mismatch_dists[test_idx])
        mismatch_proba = knn_mismatch.predict_proba(mismatch_dists[test_idx])
        calculate_acc_auroc_f1('Mismatch', results, test_labels, mismatch_pred, mismatch_proba)

        mix_labels = kernel_mix_transform_labels(labels, train_idx)
        flat_dists = flat_dists_calc(test_points, train_points)

        mix_pred_proba = mix_dists_calc(test_idx, train_idx, mix_labels, flat_dists, pair_dists, bandwidth, 0.0)
        mix_pred_labels = np.argmax(mix_pred_proba, axis=1)
        calculate_acc_auroc_f1('Mix0.00', results, test_labels, mix_pred_labels, mix_pred_proba)

        mix_pred_proba = mix_dists_calc(test_idx, train_idx, mix_labels, flat_dists, pair_dists, bandwidth, 0.25)

        mix_pred_labels = np.argmax(mix_pred_proba, axis=1)
        calculate_acc_auroc_f1('Mix0.25', results, test_labels, mix_pred_labels, mix_pred_proba)

        mix_pred_proba = mix_dists_calc(test_idx, train_idx, mix_labels, flat_dists, pair_dists, bandwidth, 0.5)
        mix_pred_labels = np.argmax(mix_pred_proba, axis=1)
        calculate_acc_auroc_f1('Mix0.50', results, test_labels, mix_pred_labels, mix_pred_proba)

        mix_pred_proba = mix_dists_calc(test_idx, train_idx, mix_labels, flat_dists, pair_dists, bandwidth, 0.75)
        mix_pred_labels = np.argmax(mix_pred_proba, axis=1)
        calculate_acc_auroc_f1('Mix0.75', results, test_labels, mix_pred_labels, mix_pred_proba)

        kernel_pred_proba = mix_dists_calc(test_idx, train_idx, mix_labels, flat_dists, pair_dists, bandwidth, 1.0)
        kernel_pred_labels = np.argmax(kernel_pred_proba, axis=1)
        calculate_acc_auroc_f1('KR', results, test_labels, kernel_pred_labels, kernel_pred_proba)

        et_predicted = extrees.predict(test_points)
        et_pred_proba = extrees.predict_proba(test_points)
        calculate_acc_auroc_f1('ExtraTrees', results, test_labels, et_predicted, et_pred_proba)

        knn_default = KNeighborsClassifier(n_neighbors=num_neighbors)
        knn_default.fit(train_points, train_labels)
        knn_pred = knn_default.predict(test_points)
        knn_pred_proba = knn_default.predict_proba(test_points)
        calculate_acc_auroc_f1('KNN', results, test_labels, knn_pred, knn_pred_proba)
    return results


def to_metric_model_dataframe(results: dict[dict[str, float]]):
    metrics = ["Accuracy", "AUROC", "F1"]
    models = ["Metric", "Mismatch", "Mix0.00", "Mix0.25", "Mix0.50", "Mix0.75", 
              "KR", "ExtraTrees", "KNN"]
    columns = pd.MultiIndex.from_product([metrics, models], names=["Metric", "Model"])
    data = {
        (metric, model): results[metric][model]
        for metric in metrics
        for model in models
    }
    return pd.DataFrame(data, columns=columns)


# === Predictor ===

class ExtraTreesNWRegression:
    def __init__(self, alpha: float = 0.5, n_estimators: int = 25, bandwidth: float = 1.0):
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.bandwidth = bandwidth

    def fit(self, X_train, y_train):
        self.extrees = fitted_extrees(X_train, y_train, self.n_estimators)
        self.X_train_ = X_train
        self.y_train_ = y_train
        return self

    def predict_proba(self, X_pred):
        X_all = np.concatenate((self.X_train_, X_pred), axis=0)
        train_idx = np.arange(len(self.X_train_))
        pred_idx = np.arange(len(X_pred)) + len(self.X_train_)
        pair_dists = leaf_precomp_no_cdist_pairwise_distances_with_padded_no_cdist_leaves(self.extrees, X_all, train_idx)
        flat_dists = gaussian_dist_flat_bc(X_pred, self.X_train_)
        mix_labels = kernel_mix_transform_labels(self.y_train_, np.arange(len(self.y_train_)))
        return predictions_from_flat_and_pairdist_mix_tree_unsquared(pred_idx, train_idx, mix_labels, flat_dists, pair_dists, self.bandwidth, self.alpha)

    def predict(self, X_pred):
        return np.argmax(self.predict_proba(X_pred), axis=1)


# === Wisconsin breast cancer ===

def dataset_wisconsin_breast_cancer():
    dataset_id = 15  # Breast Cancer (Wisconsin)
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    return X, y


def wisconsin_breast_cancer_prepared(X, y, train_indices: np.ndarray):
    X_train = X.iloc[train_indices]
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)
    scaler = StandardScaler()
    X_imputed = imputer.transform(X)
    scaler.fit(imputer.transform(X_train))
    X_scaled = scaler.transform(X_imputed)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X_scaled, y_encoded

# === Summary functions ===

def csv_name(
    label: str, n_estimators: int, num_neighbors: int, test_swap: bool,
    n_splits: int, n_repeats: int, alpha: float, bandwidth: float
):
    return f'./data/{label}_{time.time_ns()}_est{n_estimators}_k{num_neighbors}_{"swapped" if test_swap else "unswapped"}_sp{n_splits}_rp{n_repeats}_alpha{alpha}_bw{bandwidth}.csv'


def save_and_print_results(results: dict, label: str):
    results_df = to_metric_model_dataframe(results)
    file_name = csv_name(label, n_estimators, num_neighbors, test_swap, n_splits, n_repeats, alpha, bandwidth)
    results_df.to_csv(file_name, index=False)
    print(f'{label.capitalize()}:')
    print(results_df.mean() * 100)
    return file_name, results_df


def stratified_subsample(X, y, sample_size, random_state=42):
    if sample_size == len(X):
        return X, y
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=sample_size, stratify=y, random_state=random_state
    )
    return X_sample, y_sample


# === Diabetic Retinopathy ===

def dataset_diabetic_retinopathy():
    dataset_id = 37  # 43341 40666
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    return X, y


def diabetic_retinopathy_prepared(X, y, train_indices: np.ndarray):
    X_train = X.iloc[train_indices]
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)  # Fit only on the training data
    scaler = StandardScaler()
    X_imputed = imputer.transform(X)
    scaler.fit(imputer.transform(X_train))
    X_scaled = scaler.transform(X_imputed)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X_scaled, y_encoded


# === EEG Eyes ===

def dataset_eeg_eyes():
    dataset_id = 1471  # Dataset ID for EEG Eyes
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    subsample_size = 0.5
    X, y = stratified_subsample(X, y, int(len(X) * subsample_size), random_state=42)
    return X, y


def eeg_eyes_prepared(X, y, train_indices: np.ndarray):
    X_train = X.iloc[train_indices]
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)  # Fit only on the training data
    scaler = StandardScaler()
    X_imputed = imputer.transform(X)
    scaler.fit(imputer.transform(X_train))
    X_scaled = scaler.transform(X_imputed)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.to_numpy())
    return X_scaled, y_encoded


# === Haberman ===

def dataset_haberman():
    dataset_id = 43  # Dataset ID for Haberman
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    subsample_size = 1.0
    X, y = stratified_subsample(X, y, int(len(X) * subsample_size), random_state=42)
    return X, y


def haberman_prepared(X, y, train_indices: np.ndarray):
    X_train = X.iloc[train_indices]
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)  # Fit only on the training data
    scaler = StandardScaler()
    X_imputed = imputer.transform(X)
    scaler.fit(imputer.transform(X_train))
    X_scaled = scaler.transform(X_imputed)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X_scaled, y_encoded


# === Ionosphere ===

def dataset_ionosphere():
    dataset_id = 59  # Dataset ID for Ionosphere
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    subsample_size = 1.0
    X, y = stratified_subsample(X, y, int(len(X) * subsample_size), random_state=42)
    return X, y


def ionosphere_prepared(X, y, train_indices: np.ndarray):
    X_train = X.iloc[train_indices]
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)  # Fit only on the training data
    scaler = StandardScaler()
    X_imputed = imputer.transform(X)
    scaler.fit(imputer.transform(X_train))
    X_scaled = scaler.transform(X_imputed)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X_scaled, y_encoded


# === Tic-Tac-Toe ===

def dataset_tic_tac_toe():
    dataset_id = 50  # Dataset ID for Tic-Tac-Toe
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    subsample_size = 1.0
    X, y = stratified_subsample(X, y, int(len(X) * subsample_size), random_state=42)
    return X, y


def tic_tac_toe_prepared(X, y, train_indices: np.ndarray):
    X_train = X.iloc[train_indices]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train)  # Fit only on the training data
    X_encoded = encoder.transform(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X_encoded, y_encoded


# === Website Phishing ===

def dataset_website_phishing():
    dataset_id = 4534  # Dataset ID for Website Phishing
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    
    subsample_size = 0.1
    X, y = stratified_subsample(X, y, int(len(X) * subsample_size), random_state=42)
    return X, y


def website_phishing_prepared(X, y, train_indices: np.ndarray):
    X_train = X.iloc[train_indices]
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_scaled = scaler.transform(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X_scaled, y_encoded


# === Dimensional Spirals ===

def dataset_dim_spirals():
    points, labels = dim_spirals(500, 8, std_dev=0.5)
    return points, labels


def dim_spirals_prepared(X, y, train_indices: np.ndarray):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded


BUILTIN_DATASETS = {
    "wisconsin": "Wisconsin Breast Cancer",
    "retinopathy": "Diabetic Retinopathy",
    "eeg_eyes": "EEG Eyes",
    "haberman": "Haberman",
    "ionosphere": "Ionosphere",
    "tic_tac_toe": "Tic-Tac-Toe",
    "website_phishing": "Website Phishing",
    "dim_spirals": "Dimensional Spirals",
}


def list_datasets():
    print("Встроенные тестовые датасеты:")
    for name, desc in BUILTIN_DATASETS.items():
        print(f"  {name} - {desc}")


def _load_data(path: str, delimiter: str = ",", quotechar: str = '"'):
    df = pd.read_csv(path, delimiter=delimiter, quotechar=quotechar)
    return df.values


def cli():
    parser = ArgumentParser("XTNW")
    parser.add_argument('x_train', type=str, help='Training features file path (CSV).')
    parser.add_argument('y_train', type=str, help='Target values file path (CSV, single column).')

    parser.add_argument('--x_pred', type=str, default=None, help='Input features to predict on file path (CSV).')
    parser.add_argument('--pred_output', type=str, default=None, help='Predictions output file (defaults to stdout).')

    parser.add_argument('--delimiter', type=str, default=',', help='CSV delimiter.')
    parser.add_argument('--quotechar', type=str, default='"', help='CSV quote char.')

    parser.add_argument('--n_estimators', type=int, default=25,
                        help='Number of estimators (trees) in the forest, >=1 (default: 25).')
    parser.add_argument('--bandwidth', type=float, default=1.0,
                        help='Bandwidth (smoothing) >0 (default: 1.0).')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Mixing coefficient between RBF and ExtraTrees kernel, in [0,1] (default: 0.5).')

    args = parser.parse_args()

    if args.n_estimators < 1:
        parser.error("n_estimators must be >= 1")
    if args.bandwidth <= 0:
        parser.error("bandwidth must be > 0")
    if not (0.0 <= args.alpha <= 1.0):
        parser.error("alpha must be between 0 and 1 (inclusive)")

    loader = partial(_load_data, delimiter=args.delimiter, quotechar=args.quotechar)

    try:
        X_train = loader(args.x_train)
    except Exception as e:
        print(f"Error loading X_train from {args.x_train}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        y_train = loader(args.y_train)
    except Exception as e:
        print(f"Error loading y_train from {args.y_train}: {e}", file=sys.stderr)
        sys.exit(2)

    n_samples, n_features = X_train.shape
    n_target_samples, n_target_features = y_train.shape
    assert n_samples == n_target_samples, 'Number of training samples and target values should be the same'
    assert n_target_features == 1, 'Number of target features should be equal to one'

    y_train = y_train.squeeze(1)
    print(f"Loaded {n_samples} training samples with {n_features} features.")

    time_stamp = time.time()
    model = ExtraTreesNWRegression(args.alpha, args.n_estimators, args.bandwidth)
    model.fit(X_train, y_train)
    fit_time = time.time() - time_stamp
    print('Training time:', round(fit_time, 4), 's.')


    if args.x_pred is not None:
        try:
            X_pred = loader(args.x_pred)
        except Exception as e:
            print(f"Error loading X_pred from {args.x_pred}: {e}", file=sys.stderr)
            sys.exit(3)

        if X_pred.ndim != 2 or X_pred.shape[1] != n_features:
            raise AssertionError('Number of features to predict on should be equal to the number of training features')

        predictions = model.predict(X_pred)
        
        np.savetxt(args.pred_output or sys.stdout, predictions[:, np.newaxis])


if __name__ == '__main__':
    cli()
