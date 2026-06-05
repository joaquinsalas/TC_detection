# clasificador SVM que identifica por el num de trayectorias y la dispersion cuando un ciclón es maduro o no
#con label confirmed
# Entrena 30 clasificadores, escoje el mejor y lo guarda como Best_XGB_classiffier_seed_{best_seed}.pkl
# Los otros 30 clasificadores tambien los guarda para hacer con ellos un ensamble
# En este archivo solo se hace el entrenamiento 
#

#modificar para que use la cpu en caso de no encontrar GPU

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import optuna
from sklearn.svm import SVC
from classifier.train_common import split_and_preprocess, calcula_PR_ascendente, count_dir
from common import reset_output_paths

def entrenar_modelo_svm(X, y, C, kernel, gamma, class_weight, probability):
    clf = SVC(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight, probability=probability, random_state=0)
    clf.fit(X, y)
    return clf

# -------------- Optuna Objective -------------
def objective(trial, X_train, y_train, X_val, y_val):
    C = trial.suggest_loguniform('C', 1e-2, 100)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e-1)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    probability = True  # para poder usar predict_proba

    clf = entrenar_modelo_svm(
        X_train, y_train,
        C, kernel, gamma, class_weight, probability
    )
    probs = clf.predict_proba(X_val)[:, 1]
    auc_pr, _, _ = calcula_PR_ascendente(y_val, probs)
    return auc_pr


# main ------------------------------------------------------------------------
def main(train_n):
    csv_path = "database_creation/confirmed_umbrales_ciclones.csv"
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    out_dir = "classifier/modelos/SVM"
    best_dir = 'classifier/train/SVM/best'
    reset_output_paths(dirs=[out_dir, best_dir])

    best_score_pr = -np.inf
    best_model = None
    best_scaler = None
    best_seed = None
    best_params = None
    best_params_best = None
    params_list = []

    last_seed = count_dir(out_dir)
    seeds_list = list(range(last_seed, train_n))
    for seed in seeds_list:
        print(f"\n=== SEMILLA {seed} ===")
        # Train-test split
        X_train, X_val, X_test, y_train, y_val, y_test = split_and_preprocess(
            df,
            date_col   ='fecha_prediccion',
            train_start='2022-01-01',
            train_end  ='2023-12-31',
            val_start  ='2024-01-01',
            val_end    ='2024-12-31',
            test_start ='2025-01-01',
            test_end   ='2025-12-31',
            label      ='label',
            seed       =seed
        )

        # Escalado
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train)
        X_val_np  = scaler.transform(X_val)
        X_test_np = scaler.transform(X_test)

        X_train_arr = np.array(X_train_np)
        y_train_arr = np.array(y_train)
        X_val_arr = np.array(X_val_np)
        y_val_arr = np.array(y_val)
        # Optuna hyperparameter tuning con validation 2024
        def optuna_objective(trial):
            return objective(trial, X_train_arr, y_train_arr, X_val_arr, y_val_arr)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(optuna_objective, n_trials=2000, show_progress_bar=True)

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
    if best_params_best is not None:
        with open(os.path.join(best_dir, f'Best_SVM_classifier_seed_{best_seed}.pkl'), "wb") as f:
            pickle.dump({"scaler": best_scaler, "model": best_model, "params": best_params_best}, f)
        print(f"\nMejor modelo SVM-Optuna: seed={best_seed} con AUC-PR={best_score_pr:.3f}")

        df_params = pd.DataFrame(params_list)
        df_params.to_csv(os.path.join(best_dir, 'best_params.csv'), index=False)

if __name__ == "__main__":
    main(train_n)
