#pip install autogluon
#falta comentar en inglés

import os
import random
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from train.NN.NN_30_train import FlexibleNN
from train_common import split_and_preprocess, count_dir

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

# main ------------------------------------------------------------------------
def main(train_n):
    MODELS_DIR = "classifier/modelos"
    csv_path = "database_creation/confirmed_umbrales_ciclones.csv"
    OUT_DIR = "classifier/modelos/Gluon"
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    #semillas=[2]
    #for seed in semillas:

    last_seed = count_dir(OUT_DIR)
    seeds_list = list(range(last_seed, train_n))
    for seed in seeds_list:
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

    print("\nSe han entrenado y guardado los 30 ensambles.")


if __name__ == "__main__":
    main(train_n)