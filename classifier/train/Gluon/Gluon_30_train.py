#pip install autogluon


import os
import random
import glob
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
from autogluon.tabular import TabularPredictor
from sklearn.utils import shuffle

# --- Red neuronal en PyTorch ---
class FlexibleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, activation_name):
        super().__init__()
        layers = []
        prev_dim = input_dim
        # Elige activación según nombre
        if activation_name == "relu":
            activation = nn.ReLU()
        elif activation_name == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name == "tanh":
            activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation_name}")

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # output
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def nn_bundle_predict_proba(nn_bundle, X_test):
    """nn_bundle: dict con keys 'scaler', 'model_state', 'params'."""
    scaler = nn_bundle["scaler"]
    state_dict = nn_bundle["model_state"]
    params = nn_bundle["params"]

    # Asegura numpy 2D
    X_np = X_test.values if hasattr(X_test, "values") else X_test
    Xs = scaler.transform(X_np)  # usa el scaler guardado en el bundle

    # Reconstruir la arquitectura exacta
    input_dim = Xs.shape[1]
    n_layers = params.get("n_layers", 1)
    hidden_dims = [params.get(f"n_units_layer_{i}", 64) for i in range(n_layers)]
    dropout = params.get("dropout", 0.0)
    activation_name = params.get("activation", "relu")

    model = FlexibleNN(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, activation_name=activation_name)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(Xs, dtype=torch.float32)).squeeze(1)
        probas = torch.sigmoid(logits).cpu().numpy()
    return probas

def svm_bundle_predict_proba(svm_bundle, X_test):
    scaler = svm_bundle["scaler"]
    model = svm_bundle["model"]

    # Asegura numpy 2D
    X_np = X_test.values if hasattr(X_test, "values") else X_test
    Xs = scaler.transform(X_np)
    probas = model.predict_proba(Xs)[:, 1]
    return probas

# --- Helper para cargar modelos y obtener probas ---
def predict_with_model(model_info, X):
    # Si es un bundle NN (dict), reconstruimos; si no, asumimos sklearn (XGB Pipeline)
    if isinstance(model_info, dict) and "model_state" in model_info and "scaler" in model_info:
        probas = nn_bundle_predict_proba(model_info, X)
    # Si es un bundle SVM
    elif isinstance(model_info, dict) and "model" in model_info and "scaler" in model_info:
        probas = svm_bundle_predict_proba(model_info, X)
    # sirve para XGB
    elif "sklearn.pipeline" in str(type(model_info)):
        # predict_proba(X) ya incluye el escalado
        probas = model_info.predict_proba(X)[:, 1]
    else:
        print('tipo de modelo no reconocido')
        probas = None
    return probas


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

# main ------------------------------------------------------------------------
def main():
    MODELS_DIR = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos"
    csv_path = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv"
    OUT_DIR = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos/Gluon"
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    best_score_pr = -np.inf
    best_model = None
    best_scaler = None
    best_seed = None
    best_params = None


    semillas=[2]
    #for seed in range(30):
    for seed in semillas:
        print(f"\n=== SEMILLA {seed} ===")
        X_train, X_test, y_train, y_test = split_and_preprocess(
            df, 'fecha_prediccion', '2023-01-01', '2024-12-31', 'label', seed
        )

        # 1. Selecciona 3 modelos de cada tipo al azar
        NN_files  = glob.glob(os.path.join(f'{MODELS_DIR}/NN', "NN_classifier_seed_*.pkl"))
        XGB_files = glob.glob(os.path.join(f'{MODELS_DIR}/XGB', "XGB_classiffier_seed_*.pkl"))
        SVM_files = glob.glob(os.path.join(f'{MODELS_DIR}/SVM', "SVM_classifier_seed_*.pkl"))
        NN_sel  = np.random.RandomState(seed).choice(NN_files,  3, replace=False)
        XGB_sel = np.random.RandomState(seed).choice(XGB_files, 3, replace=False)
        SVM_sel = np.random.RandomState(seed).choice(SVM_files, 3, replace=False)
        selected_files = list(NN_sel) + list(XGB_sel) + list(SVM_sel)
        #np.random.shuffle(selected_files)  # Mezcla el orden

        # 2. Genera las features de stacking: proba de cada modelo
        train_preds = pd.DataFrame()
        test_preds  = pd.DataFrame()
        scalers = []

        for idx, f in enumerate(selected_files):
            with open(f, "rb") as pf:
                model_info = pickle.load(pf)
            col_name = f"model_{idx}"
            train_preds[col_name] = predict_with_model(model_info, X_train)
            test_preds[col_name]  = predict_with_model(model_info, X_test)
            scalers.append(model_info['scaler'])

        # Añade la columna objetivo para entrenamiento AutoGluon
        train_preds['label'] = y_train.values
        test_preds['label']  = y_test.values

        # 3. Entrena el ensamblador (AutoGluon)
        save_path = os.path.join(OUT_DIR, f"ensemble_seed_{seed}")
        predictor = TabularPredictor(label='label', problem_type='binary', eval_metric='average_precision',
                                    path=save_path).fit(
            train_data=train_preds,
            presets="best_quality",
            time_limit=60*10  # 10 minutos
        )

        # Evalúa
        perf = predictor.evaluate(test_preds)
        print(f"AUC-PR (test): {perf['average_precision']:.3f}")

        # 4. Guarda info extra (scalers y nombres de modelos usados)
        meta = {
            "scalers": scalers,
            "model_files": selected_files
        }
        with open(os.path.join(save_path, f"ensemble_meta_{seed}.pkl"), "wb") as mf:
            pickle.dump(meta, mf)

    print("\n¡Listo! Se han entrenado y guardado los 30 ensambles.")


if __name__ == "__main__":
    main()