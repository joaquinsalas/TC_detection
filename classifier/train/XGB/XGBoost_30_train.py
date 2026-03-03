# clasificador XGBoosting que identifica por el num de trayectorias y la dispersion cuando un ciclón es maduro o no
#con label confirmed
# Entrena 30 clasificadores, escoje el mejor y lo guarda como Best_XGB_classiffier_seed_{best_seed}.pkl
# Los otros 30 clasificadores tambien los guarda para hacer con ellos un ensamble
# En este archivo solo se hace el entrenamiento 

#Nota: a este se le agregó también la colmna de zona
#nota: quitale la columna de zona, es mejor

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve, auc
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from sklearn.pipeline import Pipeline
import os, glob

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

def refit_with_early_stopping(best_params, X_train, y_train, random_state):
    # 1) Construye el pipeline y aplica los mejores params
    pipe = build_pipeline(random_state=random_state).set_params(**best_params)

    # 2) Split interno para early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # 3) Fit SOLO el scaler con train y transforma train/val
    scaler = pipe.named_steps["scaler"]
    xgb    = pipe.named_steps["xgb"]

    scaler.fit(X_tr)                    # <- evita fuga de datos
    X_tr_s  = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # 4) Early stopping en el estimador (ponlo como parámetro, no en fit, para evitar el warning)
    xgb.set_params(n_estimators=2000, early_stopping_rounds=10)  # grande; se detiene solo #antes era un 100
    xgb.fit(
        X_tr_s, y_tr,
        eval_set=[(X_val_s, y_val)],
        verbose=False
    )

    # 5) Devuelve el pipeline con scaler y xgb YA entrenados
    return pipe


def preprocess(df, label, flag, seed = None):
    if flag =='train':
        # Balancear clases y normalizar X
        min_class = df[label].value_counts().min()
        df_0 = df[df[label] == 0].sample(min_class, random_state=seed)
        df_1 = df[df[label] == 1].sample(min_class, random_state=seed)
        df_bal = pd.concat([df_0, df_1]).sample(frac=1, random_state=seed)
    elif flag == 'test':
        df_bal = df

    # zona: pacifico=1, atlantico=0
    #df_bal['zona'] = np.where(df_bal['zona'] == 'pacifico', 1, 0)
    X = df_bal[["n_trayectorias_best_cluster", "dispersión_km_best_cluster",'horas_diff_estimadas']] #'zona'
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


def tune_with_random(X_train, y_train, random_state):
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
    
    rand = RandomizedSearchCV(
        estimator           = pipe,
        param_distributions = param_dist,
        n_iter              = 200,
        scoring             = "average_precision", #'roc_auc'
        cv                  = 10,
        verbose             = 1,
        n_jobs              = -1,
        random_state        = random_state
    )
    rand.fit(X_train, y_train)
    print("Mejor AUC (train CV):", rand.best_score_)
    print("Mejores parámetros:", rand.best_params_)
    return rand.best_estimator_, rand.best_params_

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
    # Leer el CSV
    csv_path = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv"
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    # el clasificador se entrena 30 veces con particiones difeerentes y nos quedamos con el mejor
    records = []            # para acumular resultados
    best_score_roc = -np.inf
    best_score_pr = -np.inf
    best_clf   = None
    best_seed  = None
    seeds_list = list(range(30))
    for seed in seeds_list:
        # Train-test split
        X_train, X_test, y_train, y_test = split_and_preprocess(
            df,
            date_col='fecha_prediccion',
            train_start='2023-01-01',
            train_end  ='2024-12-31',
            label='label',
            seed = seed
        )

        # 2) Optimización de hiperparámetros
        _, params = tune_with_random(X_train, y_train, seed)
        clf = refit_with_early_stopping(params, X_train, y_train, seed)

        #guarda los clasificadores
        with open(f"/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos/XGB/XGB_classiffier_seed_{seed}.pkl", "wb") as f:
            pickle.dump(clf, f)
    
        # 2.3) Predicción y métricas
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr, _, _ = calcula_PR_ascendente(y_test, y_proba)
        
        # 2.4) Guardar CSV individual
        eval_df = pd.DataFrame({
            'seed':      [seed],
            'auc_roc':   [auc_roc],
            'auc_pr':    [auc_pr]
        })
        eval_df.to_csv(f"/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/XGB/evaluaciones/evaluation_seed_{seed}.csv", index=False)

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

     # 3) Guardar resumen general en Excel
    results_df = pd.DataFrame(records)
    results_df.to_csv(
        "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/XGB/best_params.csv",
        index=False
    )

    # 4) Guardar mejor clasificador
    with open(f"/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/XGB/Best_XGB_classiffier_seed_{best_seed}.pkl", "wb") as f:
        pickle.dump(best_clf, f)
    print(f"Mejor modelo: seed={best_seed} con AUC-ROC={best_score_roc:.3f} y AUC-PR={best_score_pr:.3f}")


if __name__ == "__main__":
    main()