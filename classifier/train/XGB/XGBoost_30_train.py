# clasificador XGBoosting que identifica por el num de trayectorias y la dispersion cuando un ciclón es maduro o no
#con label confirmed
# Entrena 30 clasificadores, escoje el mejor y lo guarda como Best_XGB_classifier_seed_{best_seed}.pkl
# Los otros 30 clasificadores tambien los guarda para hacer con ellos un ensamble
# En este archivo solo se hace el entrenamiento 

#Nota: a este se le agregó también la colmna de zona
#nota: quitale la columna de zona, es mejor
#

#modificar para que no se ejecute el entrenamiento si es que ya están los 30 modelos entrenados
#modifcar para que empiece a entrenar desde el modelo falttane (si es que ya había otros)
#falta comentar a inglés

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
import scipy.stats as stats
from sklearn.pipeline import Pipeline
import os
from classifier.train_common import split_and_preprocess, calcula_PR_ascendente, count_dir
from common import reset_output_paths

#definición de métodos
def build_pipeline(random_state=42):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            subsample=1.0,
            colsample_bytree=1.0
        ))
    ])
    return pipe

def refit_with_early_stopping(best_params, X_train, y_train, X_val, y_val, random_state):
    # Construye el pipeline y aplica los mejores params
    pipe = build_pipeline(random_state=random_state).set_params(**best_params)

    scaler = pipe.named_steps["scaler"]
    xgb    = pipe.named_steps["xgb"]
    #Transformar train y validation con el mismo scaler
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # Entrenar XGB usando validation 2024 para early stopping
    xgb.set_params(n_estimators=2000, early_stopping_rounds=10)
    xgb.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=False
    )
    # Devuelve el pipeline con scaler y xgb ya entrenados
    return pipe


def tune_with_random(X_train, y_train, X_val, y_val, random_state):
    pipe = build_pipeline(random_state=random_state)
    param_dist = {
        'xgb__n_estimators':      stats.randint(50, 300),
        'xgb__max_depth':         stats.randint(2, 10),
        'xgb__learning_rate':     stats.uniform(0.001, 0.3),
        'xgb__gamma':             stats.uniform(0, 5),
        "xgb__min_child_weight":  stats.randint(1, 8),
        "xgb__reg_alpha":         stats.uniform(0.0, 1.0),
        "xgb__reg_lambda":        stats.uniform(1.0, 5.0),
    }
    
    # Juntar train + validation solo para que PredefinedSplit sepa
    # qué filas son train y cuáles son validation.
    X_search = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_search = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    # -1 = siempre entrenamiento
    #  0 = validation fold
    test_fold = [-1] * len(X_train) + [0] * len(X_val)
    ps = PredefinedSplit(test_fold=test_fold)

    rand = RandomizedSearchCV(
        estimator           = pipe,
        param_distributions = param_dist,
        n_iter              = 2000,
        scoring             = "average_precision", #'roc_auc'
        cv                  = ps,
        verbose             = 1,
        n_jobs              = -1,
        random_state        = random_state,
        refit=False
    )
    rand.fit(X_search, y_search)
    print("Mejor AUC-PR en validation 2024:", rand.best_score_)
    print("Mejores parámetros:", rand.best_params_)
    return rand.best_params_

# main ------------------------------------------------------------------------
def main(train_n):
    # Leer el CSV
    csv_path = "database_creation/confirmed_umbrales_ciclones.csv"
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    out_dir = "classifier/modelos/XGB"
    best_dir = 'classifier/train/XGB/best'
    reset_output_paths(dirs=[out_dir, best_dir])

    # el clasificador se entrena 30 veces con particiones difeerentes y nos quedamos con el mejor
    records = []            # para acumular resultados
    best_score_roc = -np.inf
    best_score_pr = -np.inf
    best_clf   = None
    best_seed  = None

    last_seed = count_dir(out_dir)
    seeds_list = list(range(last_seed, train_n))
    for seed in seeds_list:
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

        # Optimización de hiperparámetros
        params = tune_with_random(X_train, y_train, X_val, y_val, seed)
        # Entrenamiento final
        clf = refit_with_early_stopping(params, X_train, y_train, X_val, y_val, seed)

        #guarda los clasificadores
        with open(os.path.join(out_dir, f'XGB_classifier_seed_{seed}.pkl'), "wb") as f:
            pickle.dump(clf, f)
    
        # 2.3) Predicción y métricas
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr, _, _ = calcula_PR_ascendente(y_test, y_proba)

        # 2.5) Acumular y checar mejor
        if auc_pr > best_score_pr:
            best_score_roc = auc_roc
            best_score_pr = auc_pr
            best_clf   = clf
            best_seed  = seed

        # Guardar todos los resultados y parámetros
        record = {'seed': seed, 'auc_roc': auc_roc, 'auc_pr': auc_pr}
        record.update(params)  # añade los hiperparámetros
        records.append(record)

    # 4) Guardar mejor clasificador
    if best_seed is not None:
        with open(os.path.join(best_dir, f'Best_XGB_classifier_seed_{best_seed}.pkl'), "wb") as f:
            pickle.dump(best_clf, f)
        print(f"Mejor modelo XGB: seed={best_seed} con AUC-ROC={best_score_roc:.3f} y AUC-PR={best_score_pr:.3f}")

        results_df = pd.DataFrame(records)
        results_df.to_csv(os.path.join(best_dir, f'best_params.csv'), index=False)

if __name__ == "__main__":
    main(train_n)
