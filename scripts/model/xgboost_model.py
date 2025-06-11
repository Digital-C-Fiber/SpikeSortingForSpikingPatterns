import optuna
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from scripts.helper import *
from snakemake.script import snakemake
from optuna.exceptions import TrialPruned

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

# evaluate xgboost classifer
def evaluate_classifier(pred_labels, true_labels, track_of_interest):
    #pred_labels = classified_df[pred_col]
    TP = ((true_labels == track_of_interest) & (pred_labels == track_of_interest)).sum()
    FP = ((true_labels != track_of_interest) & (pred_labels == track_of_interest)).sum()
    FN = ((true_labels == track_of_interest) & (pred_labels != track_of_interest)).sum()
    TN = ((true_labels != track_of_interest) & (pred_labels != track_of_interest)).sum()
    accuracy = (TP + TN) / len(true_labels)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1_score
    }

# optuna objective to find best hyperparameters 
def objective_xgb(trial, features_all, spikes_all, ap_track_all):
    max_depth = trial.suggest_int('max_depth', 3, 10)
    gamma = trial.suggest_float('gamma', 0.0, 5.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0.0, 50.0)
    reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    
    # Fixed seed for reproducibility
    seed = 0
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    track_names = list(ap_track_all["track"].unique())
    scores = []
    for train_idx,test_idx in kf.split(features_all):
        #display(spikes_raw_background_svc)
        df_train_fold = features_all.iloc[train_idx]
        X_train_fold = df_train_fold.dropna(axis=1)
        y_train_fold = ap_track_all.iloc[train_idx]["track"].astype("category")
        params = {
        'max_depth': max_depth,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': min_child_weight,
        'n_estimators': n_estimators,
        'seed': seed, 
        'enable_categorical':True,
        }
        num_classes = len(track_names)

        if num_classes == 2:
            params['objective'] = 'binary:logistic'
        else:
            params['objective'] = 'multi:softmax'
            params['num_class'] = num_classes
        
        clf_xgb = xgb.XGBClassifier(**params)
        le = LabelEncoder()
        clf_xgb.fit(X_train_fold, le.fit_transform(y_train_fold))
        X_test_fold = features_all.iloc[test_idx]
        y_test_fold = ap_track_all.iloc[test_idx]["track"]
        predictions = clf_xgb.predict(X_test_fold)
        scores.append(f1_score(le.inverse_transform(predictions), y_test_fold.values,  average="macro"))   

    # average score for all 5-folds on background spikes
    avg_score = np.mean(scores)
    trial.set_user_attr("cv_scores", scores)
    return avg_score

# save objective for xgb
def safe_objective_xgb(trial):
    try:
        return objective_xgb(
            trial,
            spikes_all=spikes_all,
            features_all=features_background_all,
            ap_track_all=ap_track_window_a
        )
    except Exception as e:
        print(f"XGB Trial failed with exception: {e}")
        raise TrialPruned(f"Pruned trial due to exception: {e}")
    
# create optuna study
study_xgb = optuna.create_study(
    study_name=f"xgb_cv_{dataset_name}_f1",
    #load_if_exists=True,
    direction="maximize"
)

# optimzize objective and save best params and model
study_xgb.optimize(safe_objective_xgb, n_trials=20, n_jobs=1, timeout=60)
best_params_xgb = study_xgb.best_params
best_cv_scores_xgb = study_xgb.best_trial.value
best_model_xgb =  xgb.XGBClassifier(**best_params_xgb)

# prepare data and fit best model
valid_idx = features_background_all.index.intersection(ap_track_window_a.index)
X_train = features_background_all.loc[valid_idx]
le = LabelEncoder()
y_train = ap_track_window_a.loc[valid_idx]["track"]
best_model_xgb.fit(X_train, le.fit_transform(y_train))

# differentiate between ground truth data and chemical data 
if ground_truth_flag:
    # evaluation is based on detected spikes, how many were correctly classified
    predictions = best_model_xgb.predict(X_detected)
    evaluation_xgb = evaluate_classifier(le.inverse_transform(predictions), y_detected, track_of_interest_label)
    df_xgb = pd.DataFrame([evaluation_xgb])
    df_xgb.to_csv(snakemake.output.scores_xgb, index=False)
else:
    # evaluation is based on background spikes, based on best model results
    df_oc = pd.DataFrame([{"F1": round(np.mean(best_cv_scores_xgb),2)}])
    df_oc.to_csv(snakemake.output.scores_xgb, index=False)

# save dataframes
df_best_params = pd.DataFrame([best_params_xgb])
df_best_params.to_csv(snakemake.output.best_params_xgb, index=False)

