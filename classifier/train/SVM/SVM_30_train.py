# clasificador SVM que identifica por el num de trayectorias y la dispersion cuando un ciclón es maduro o no
#con label confirmed
# Entrena 30 clasificadores, escoje el mejor y lo guarda como Best_XGB_classiffier_seed_{best_seed}.pkl
# Los otros 30 clasificadores tambien los guarda para hacer con ellos un ensamble
# En este archivo solo se hace el entrenamiento 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

def entrenar_modelo_svm(X, y, C, kernel, gamma, class_weight, probability):
    clf = SVC(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight, probability=probability, random_state=0)
    clf.fit(X, y)
    return clf

def preprocess(df, label, flag, seed = None):
    if flag =='train':
        # Balancear clases y normalizar X
        min_class = df[label].value_counts().min()
        df_0 = df[df[label] == 0].sample(min_class, random_state=seed)
        df_1 = df[df[label] == 1].sample(min_class, random_state=seed)
        df_bal = pd.concat([df_0, df_1]).sample(frac=1, random_state=seed)
    elif flag == 'test':
        df_bal = df

    X = df_bal[["n_trayectorias_best_cluster", "dispersión_km_best_cluster",'horas_diff_estimadas']]
    y = df_bal[label]
    return X, y
    

#divide los datos de entrenamiento (2023 y 2024) y test (2025) por fecha
def split_and_preprocess(df: pd.DataFrame, date_col: str,
                         train_start: str, train_end: str,
                         label: str, seed:int):
    df[date_col] = pd.to_datetime(df[date_col])

    # crear máscara
    start = pd.to_datetime(train_start)
    end   = pd.to_datetime(train_end)
    mask_train = df[date_col].between(start, end)

    # partir DataFrames
    df_train = df.loc[mask_train].reset_index(drop=True)
    df_test  = df.loc[~mask_train].reset_index(drop=True)
    X_train, y_train = preprocess(df_train, label, 'train', seed)
    X_test,  y_test  = preprocess(df_test,  label, 'test')
    return X_train, X_test, y_train, y_test

# -------------- Optuna Objective -------------
def objective(trial, X, y, n_folds):
    C = trial.suggest_loguniform('C', 1e-2, 100)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e-1)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    probability = True  # para poder usar predict_proba

    auc_prs = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        clf = entrenar_modelo_svm(X_tr, y_tr, C, kernel, gamma, class_weight, probability)
        probs = clf.predict_proba(X_val)[:, 1]
        auc_pr, _, _ = calcula_PR_ascendente(y_val, probs)
        auc_prs.append(auc_pr)
    return np.mean(auc_prs)


# Filtra los valores de Precision y Recall para que conforme el Recall dismiya, Precision aumente (y no decaiga)
def calcula_PR_ascendente(y_true, y_proba):
    precision0, recall0, _ = precision_recall_curve(y_true, y_proba)
    df_pr = pd.DataFrame({
        'recall':    recall0,
        'precision': precision0
    }).sort_values('recall', ascending=False).reset_index(drop=True)

    #Filtrar para que precision nunca baje
    prec_prev = 0.0
    mask = []
    for p in df_pr['precision']:
        if p >= prec_prev:
            mask.append(True)
            prec_prev = p
        else:
            mask.append(False)
    df_pr_f = df_pr[mask]

    recall_f = df_pr_f['recall'].values
    precision_f = df_pr_f['precision'].values

    #Calcular AUC-PR de la curva escalonada
    return auc(recall_f, precision_f), recall_f, precision_f

# main ------------------------------------------------------------------------
def main():
    csv_path = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv"
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    """#revisar el sid faltantae
    unique_sids = df['SID'].unique()
    # Leer SIDs del archivo
    with open('/home/nathaliealvarez/Personal/umbral_definition/best_tcs/TC_names_inside.txt') as f:
        sids = [line.strip() for line in f if line.strip()] 
    print('no pertenecen:' ,set(sids)- set(unique_sids))
    """
    out_dir = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos/SVM"
    best_dir = '/home/nathaliealvarez/Personal/Repos/TC_detection/classifier/train/SVM/best'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    best_score_pr = -np.inf
    best_model = None
    best_scaler = None
    best_seed = None
    best_params = None
    params_list = []
    n_folds = 5


    for seed in range(30):
        print(f"\n=== SEMILLA {seed} ===")
        X_train, X_test, y_train, y_test = split_and_preprocess(
            df, 'fecha_prediccion', '2023-01-01', '2024-12-31', 'label', seed
        )

        # Escalado (como siempre)
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train)
        X_test_np  = scaler.transform(X_test)

        X_arr = np.array(X_train_np)
        y_arr = np.array(y_train)
        # Optuna hyperparameter tuning
        def optuna_objective(trial):
            return objective(trial, X_arr, y_arr, n_folds)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(optuna_objective, n_trials=200, show_progress_bar=True)

        best_params = study.best_params
        print(f"Mejores parámetros Optuna:", best_params)
        # Añade info de la semilla
        params_row = best_params.copy()
        params_row['seed'] = seed
        params_list.append(params_row)
        
        # Entrenamiento final
        final_model = entrenar_modelo_svm(
            X_train_np, y_train,
            best_params['C'], best_params['kernel'], best_params['gamma'], best_params['class_weight'], True
        )

        # Guardar modelo y scaler
        with open(f"{out_dir}/SVM_classifier_seed_{seed}.pkl", "wb") as f:
            pickle.dump({"scaler": scaler, "model": final_model, "params": best_params}, f)

        # Evaluación en test
        probs = final_model.predict_proba(X_test_np)[:, 1]
        auc_roc = roc_auc_score(y_test, probs)
        auc_pr, _, _ = calcula_PR_ascendente(y_test, probs)

        if auc_pr > best_score_pr:
            best_score_pr = auc_pr
            best_model = final_model
            best_scaler = scaler
            best_seed = seed
            best_params_best = best_params

        print(f"Seed={seed}: AUC-PR={auc_pr:.3f}  AUC-ROC={auc_roc:.3f}")

    # Guardar mejor modelo
    with open(os.path.join(best_dir, f'Best_SVM_classifier_seed_{best_seed}.pkl'), "wb") as f:
        pickle.dump({"scaler": best_scaler, "model": best_model, "params": best_params_best}, f)
    print(f"\nMejor modelo SVM-Optuna: seed={best_seed} con AUC-PR={best_score_pr:.3f}")

    df_params = pd.DataFrame(params_list)
    df_params.to_csv(os.path.join(best_dir, 'best_params.csv'), index=False)

if __name__ == "__main__":
    main()