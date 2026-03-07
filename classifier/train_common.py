#Métodos comunes en el training de modelso

import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import os


#definición de métodos
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

# Cuenta la cantidad de archivos/carpetas dentro de un directorio.
def count_dir(out_dir):
    if not os.path.exists(out_dir):
        return 0
    with os.scandir(out_dir) as entries:
        count = sum(1 for _ in entries)
    return int(count)