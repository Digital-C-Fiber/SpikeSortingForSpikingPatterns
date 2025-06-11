from sklearn.svm import OneClassSVM
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import optuna 
from snakemake.script import snakemake
from scripts.helper import *
from optuna.exceptions import TrialPruned
from sklearn.preprocessing import StandardScaler

feature_name = snakemake.wildcards.feature
features_detected =  pd.read_pickle(snakemake.input.features_detected)
ap_track_detected = pd.read_pickle(snakemake.input.ap_track_window_d)
ap_raw_a = pd.read_pickle(snakemake.input.ap_raw_a)
ap_raw_d = pd.read_pickle(snakemake.input.ap_raw_d)
spikes_all = pd.read_pickle(snakemake.input.spikes_a)
features_background_all =  pd.read_pickle(snakemake.input.features_background_all)
ap_track_window_a = pd.read_pickle(snakemake.input.ap_track_window_a)
track_of_interest_label = snakemake.params.track_of_interest_label
dataset_name = snakemake.params.name
ground_truth_flag = snakemake.params.ground_truth_flag

# check if detected spikes where matched to ground truth data
def track_label_match(row):
    return track_of_interest_label if row["matched"] else "unknown"

ap_track_detected['track'] = ap_track_detected.apply(track_label_match, axis=1)
X_detected = features_detected
y_detected = ap_track_detected.loc[X_detected.index.intersection(ap_track_detected.index)]["track"]

# evaluate oc-svm classifer
def evaluate_oneclass_svm_on_fold(model, train_idx, df_train_fold, X_train_fold, spikes_all, features_all, class_of_interest):
    df_spikes_eval = spikes_all.drop(train_idx, errors="ignore")   
    # separate between main class and others  
    df_test_interest = df_spikes_eval[df_spikes_eval["track"] == class_of_interest]
    df_test_other = df_spikes_eval[df_spikes_eval["track"] != class_of_interest]
    
    # Get features
    valid_interest_idx = df_test_interest.index.intersection(features_all.index)
    valid_other_idx = df_test_other.index.intersection(features_all.index)
    X_test_interest = features_all.loc[valid_interest_idx]
    X_test_other = features_all.loc[valid_other_idx]
    
    total_samples = 0
    if X_test_interest is not None:
        preds_interest = model.predict(X_test_interest)
        tp = np.sum(preds_interest == 1)
        fn = np.sum(preds_interest == -1) 
        total_samples += len(preds_interest)
    if X_test_other is not None:
        preds_other = model.predict(X_test_other)
        tn = np.sum(preds_other == -1)
        fp = np.sum(preds_other == 1)
        total_samples += len(preds_other)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1

# evaluate final prediction 
def evaluate_oneclass_svm(predictions, series, label):
    series = np.array(series)
    tp = np.sum((predictions == 1) & (series == label))
    fn = np.sum((predictions == -1) & (series == label))
    fp = np.sum((predictions == 1) & (series != label))
    tn = np.sum((predictions == -1) & (series != label))
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }   

# optuna objective to find best hyperparameters 
def objective_oc(trial, features_single, spikes_single, features_all, spikes_all, track_of_interest_label):

    kernel = trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid'])
    degree = trial.suggest_int("degree", 2, 6)  # degrees from 2 to 5
    nu = trial.suggest_categorical("nu", [0.01, 0.05, 0.1, 0.2])
    gamma = trial.suggest_categorical("gamma", ['scale', 'auto', 0.001, 0.01, 0.1, 1.0])
    tol = trial.suggest_categorical("tol", [0.000001, 0.00001, 0.0001, 0.001, 0.1])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    class_of_interest = track_of_interest_label
    
    scores = []
    
    for train_idx, _ in kf.split(features_single):
        df_train_fold = spikes_single.iloc[train_idx]
        X_train_fold = features_single.iloc[train_idx]
        model = OneClassSVM(kernel=kernel, degree=degree, nu=nu, gamma=gamma, tol=tol)
        model.fit(X_train_fold)
        fold_score = evaluate_oneclass_svm_on_fold(model, train_idx, df_train_fold, X_train_fold, spikes_all, features_all, class_of_interest)
        scores.append(fold_score)
    avg_score = np.mean(scores)
    trial.set_user_attr("cv_scores", scores)
    return avg_score


# save objective for oc
def safe_objective_oc(trial):
    try:
        return objective_oc(
            trial,
            features_single=features_track_of_interest,
            spikes_single=spikes_track_of_interest,
            spikes_all=spikes_all,
            features_all=features_background_all,
            track_of_interest_label=track_of_interest_label
        )
    except Exception as e:
        print(f"OC-SVM Trial failed with exception: {e}")
        raise TrialPruned(f"Pruned trial due to exception: {e}")
    
# prepare data
spikes_track_of_interest = spikes_all[spikes_all["track"] == track_of_interest_label ].reset_index(drop=False)
spikes_track_of_interest = spikes_track_of_interest.rename(columns={"spike_idx":"old_spike_idx"}).pipe(rename_index, 'spike_idx')
valid_indices = spikes_track_of_interest['old_spike_idx'].isin(features_background_all.index)
features_track_of_interest = features_background_all.loc[spikes_track_of_interest['old_spike_idx'][valid_indices]]
features_track_of_interest = features_track_of_interest

# create optuna study
study_oc = optuna.create_study(
    study_name=f"oneclass_svm_cv_{dataset_name}_f1",
    direction="maximize")

# optimzize objective and save best params and model
study_oc.optimize(safe_objective_oc, n_trials=20, n_jobs=1, timeout=60)
best_params_oc = study_oc.best_params
best_cv_scores_oc = study_oc.best_trial.user_attrs["cv_scores"]
best_model_oc = OneClassSVM(kernel=best_params_oc["kernel"], degree=best_params_oc["degree"],
                        nu=best_params_oc["nu"], gamma=best_params_oc["gamma"], tol=best_params_oc["tol"])

# differentiate between ground truth data and chemical data 
if ground_truth_flag:
    # only train on single class
    best_model_oc.fit(features_track_of_interest)
    predictions = best_model_oc.predict(X_detected)
    evaluation_oc = evaluate_oneclass_svm(predictions, y_detected, track_of_interest_label)
    df_oc = pd.DataFrame([evaluation_oc])
else:
    df_oc = pd.DataFrame([{"F1": round(np.mean(best_cv_scores_oc),2)}])

# save dataframes
df_best_params = pd.DataFrame([best_params_oc])
df_best_params.to_csv(snakemake.output.best_params_oc, index=False)
df_oc.to_csv(snakemake.output.scores_oc, index=False)
    
    