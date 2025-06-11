import optuna
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from scripts.helper import *
from snakemake.script import snakemake
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

# evaluate svm classifer
def evaluate_svm(predictions, series,spikes,ap_track, label):
    series = np.array(series)
    spikes_copy = spikes.copy(deep=True)
    spikes_copy["prediction"] = predictions
    spikes_copy["true"] = series

    df_spikes = spikes_copy#.join(ap_track)
    tp = np.sum((predictions == label) & (series == label))
    fn = np.sum((predictions != label) & (series == label))
    fp = np.sum((predictions == label) & (series != label))
    tn = np.sum((predictions != label ) & (series != label))
    
    spikes_FP = df_spikes[(df_spikes["prediction"] == label) & (df_spikes["true"] != label)]
    spikes_FN = df_spikes[(df_spikes["prediction"] != label) & (df_spikes["true"] == label)]
    spikes_TP = df_spikes[(df_spikes["prediction"] == label) & (df_spikes["true"] == label)]
    spikes_TN = df_spikes[(df_spikes["prediction"] != label) & (df_spikes["true"] != label)]
    data = {"TP": spikes_TP, "FP": spikes_FP, "FN": spikes_FN, "TN": spikes_TN}
    
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
        #"data": data,
    }

# classify tracks based on SVM probability and threshold
def classify_tracks(df, prob_columns, threshold):
    df = df.copy()
    max_prob = df[prob_columns].max(axis=1)
    max_class = df[prob_columns].idxmax(axis=1)
    df['classifier_result_svc'] = max_class.where(max_prob >= threshold, 'unknown')
    return df

# optuna objective to find best hyperparameters 
def objective_svc(trial, features_all, spikes_all, ap_track_all):

    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'sigmoid'])
    C = trial.suggest_categorical('C', [0.01, 0.1,1, 10])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 0.001, 0.01, 0.1, 1])
    degree = trial.suggest_int('degree', 2, 5)
    coef0 = trial.suggest_categorical('coef0', [0.0, 0.1, 0.5, 1])
    tol = trial.suggest_float("tol", 1e-4, 1e-2, log=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    track_names = list(ap_track_all["track"].unique())
    scores = []
    for train_idx,test_idx in kf.split(features_all):
        df_train_fold = features_all.iloc[train_idx]
        X_train_fold = df_train_fold.dropna(axis=1)
        y_train_fold = ap_track_all.iloc[train_idx]["track"]

        clf_svc = svm.SVC(kernel=kernel, C=C, gamma=gamma,  tol = tol, probability=True, degree=degree, coef0=coef0)
        clf_svc.fit(X_train_fold, y_train_fold)
        X_test_fold = features_all.iloc[test_idx]
        y_test_fold = ap_track_all.iloc[test_idx]["track"]
        df_probs = pd.DataFrame(clf_svc.predict_proba(X_test_fold), columns=clf_svc.classes_)
        classified_df = classify_tracks(df_probs.copy(), track_names, 0.7).join(features_all)
        scores.append(f1_score(classified_df["classifier_result_svc"], y_test_fold.values,  average="macro"))   

    print("scores", scores)
    avg_score = np.mean(scores)
    trial.set_user_attr("cv_scores", scores)
    return avg_score

# save objective for svm
def safe_objective(trial):
    try:
        return objective_svc(trial, 
                             spikes_all=spikes_all, features_all=features_background_all, 
                             ap_track_all=ap_track_window_a)
    except Exception as e:
        print(f"Trial failed with exception: {e}")
        raise TrialPruned(f"Pruned trial due to exception: {e}")

# create optuna study
study_svm = optuna.create_study(
    study_name=f"svc_cv_{dataset_name}_f1",
    #load_if_exists=True,
    direction="maximize"
)
study_svm.optimize(safe_objective, n_trials=20, n_jobs=1, timeout=60)

# optimzize objective and save best params and model
best_params_svm = study_svm.best_params
best_cv_scores_svm = study_svm.best_trial.value
best_model_svm = svm.SVC(kernel=best_params_svm["kernel"], #degree=best_params_svm["degree"]
                        C=best_params_svm["C"], gamma=best_params_svm["gamma"], tol=best_params_svm["tol"])
                        #coef0=best_params_svm["coef0"])

# prepare data and fit best model
valid_idx = features_background_all.index.intersection(ap_track_window_a.index)
X_train = features_background_all.loc[valid_idx]
y_train = ap_track_window_a.loc[valid_idx]["track"]
best_model_svm.fit(X_train, y_train)

# differentiate between ground truth data and chemical data 
if ground_truth_flag:
    # evaluation is based on detected spikes, how many were correctly classified
    predictions = best_model_svm.predict(X_detected)
    evaluation_svm = evaluate_svm(predictions, y_detected,X_detected, ap_raw_d, track_of_interest_label)
    df_svm = pd.DataFrame([evaluation_svm])
    df_svm.to_csv(snakemake.output.scores_svm, index=False)
else:
    # evaluation is based on background spikes, based on best model results
    df_svm = pd.DataFrame([{"F1": round(np.mean(best_cv_scores_svm),2)}])
    df_svm.to_csv(snakemake.output.scores_svm, index=False)

# save dataframes
df_best_params = pd.DataFrame([best_params_svm])
df_best_params.to_csv(snakemake.output.best_params_svm, index=False)