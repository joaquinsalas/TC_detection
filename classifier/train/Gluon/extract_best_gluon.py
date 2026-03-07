#aqui se extrae la info del mejor modelo autogluon
#primero se deben evaluar todos para extraer quel que haya obtenido un AUC_PR más alto y a él se le sacarán las características

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
import os
import re
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from autogluon.tabular import TabularPredictor
import glob
from train_common import calcula_PR_ascendente, split_and_preprocess
from train.Gluon.Gluon_30_train import predict_with_model
import os
import json
import shutil




#definición de métodos
def eval_dir_gluon(base_dir, df, label, n_seeds=30):
    best_pr = 0
    best_seed = None

    y_tests, y_probas, seeds = [], [], []
    roc_scores, pr_scores = [], []

    for seed in range(n_seeds):
        ens_dir = os.path.join(base_dir, f'ensemble_seed_{seed}')
        test_preds_path = os.path.join(ens_dir, 'test_preds.csv')
        if not os.path.exists(test_preds_path):
            print(f"[Seed {seed}] test_preds.csv no existe, saltando.")
            continue
        if not os.path.exists(ens_dir):
            print(f"[Seed {seed}] Carpeta del ensamble no existe, saltando.")
            continue

        # Carga el predictor AutoGluon
        predictor = TabularPredictor.load(ens_dir)
        test_preds = pd.read_csv(test_preds_path)
        # El label real debe estar en la columna 'label'
        y_test = test_preds[label].values
        # El predictor espera solo los features como input
        X_test_features = test_preds.drop(columns=[label])

        # Predice probabilidades (de clase positiva)
        y_proba = predictor.predict_proba(X_test_features)[1]  # [1] = proba de clase positiva

        # Métricas globales
        roc = roc_auc_score(y_test, y_proba)
        pr, _, _ = calcula_PR_ascendente(y_test, y_proba)

        y_tests.append(y_test)
        y_probas.append(y_proba)
        seeds.append(seed)
        roc_scores.append(roc)
        pr_scores.append(pr)

        if pr > best_pr:
            best_pr = pr
            best_seed = seed

    return y_tests, y_probas, seeds, roc_scores, pr_scores, best_pr, best_seed

def preparar_implementation(best_autogluon_dir,seeds_dict, selected_files, best_seed, implementation_dir):
    # asegurar que implementation existe
    os.makedirs(implementation_dir, exist_ok=True)

    #guardar las seeds en el json
    json_path = os.path.join(implementation_dir, "base_models_seeds.json")
    with open(json_path, "w") as f:
        json.dump(seeds_dict, f, indent=4)

    modelos_dir = os.path.join(implementation_dir, "modelos_ensamble")
    # crear o vaciar modelos_ensamble
    if os.path.exists(modelos_dir):
        for sub in os.listdir(modelos_dir):
            sub_path = os.path.join(modelos_dir, sub)
            if os.path.isfile(sub_path) or os.path.islink(sub_path):
                os.unlink(sub_path)
            elif os.path.isdir(sub_path):
                shutil.rmtree(sub_path)
    else:
        os.makedirs(modelos_dir)

    # copiar los modelos seleccionados
    for src in selected_files:
        filename = os.path.basename(src)
        dest = os.path.join(modelos_dir, filename)
        shutil.copy2(src, dest)

    # borrar carpetas ensemble_seed_*
    for item in os.listdir(implementation_dir):
        path = os.path.join(implementation_dir, item)
        if item.startswith("ensemble_seed_") and os.path.isdir(path):
            shutil.rmtree(path)

    # copiar el mejor ensemble de AutoGluon
    destination = os.path.join(implementation_dir, f"ensemble_seed_{best_seed}")
    if os.path.exists(destination):
        shutil.rmtree(destination)
        classifier/modelos/Gluon
    shutil.copytree(best_autogluon_dir, destination)

def extract_seed(filepath):
    """Extrae el número de seed del nombre del archivo."""
    filename = os.path.basename(filepath)
    match = re.search(r'seed_(\d+)', filename)
    return int(match.group(1)) if match else None

# Selecciona 3 modelos al azar de cada tipo con la misma semilla
def select_rand_models(MODELS_DIR, seed):
    # la random seed hace que siemore sean los mismos modelos individuales
    NN_files  = glob.glob(os.path.join(f'{MODELS_DIR}/NN', "NN_classifier_seed_*.pkl"))
    XGB_files = glob.glob(os.path.join(f'{MODELS_DIR}/XGB', "XGB_classifier_seed_*.pkl"))
    SVM_files = glob.glob(os.path.join(f'{MODELS_DIR}/SVM', "SVM_classifier_seed_*.pkl"))
    NN_sel  = np.random.RandomState(seed).choice(NN_files,  3, replace=False)
    XGB_sel = np.random.RandomState(seed).choice(XGB_files, 3, replace=False)
    SVM_sel = np.random.RandomState(seed).choice(SVM_files, 3, replace=False)
    selected_files = list(NN_sel) + list(XGB_sel) + list(SVM_sel)
    #np.random.shuffle(selected_files)  # Si NO mezclaste durante el entrenamiento, no mezcles aquí
    return selected_files, NN_sel, XGB_sel, SVM_sel

def generate_test_preds_for_gluon_ensembles(
    df, MODELS_DIR, ENSEMBLES_DIR, split_and_preprocess, predict_with_model,
    label='label', n_seeds=30):

    for seed in range(n_seeds):
        print(f"\n=== SEMILLA {seed} ===")
        # 1. Split como en entrenamiento
        _, X_test, _, y_test = split_and_preprocess(
            df, 'fecha_prediccion', '2023-01-01', '2024-12-31', label, seed
        )

        selected_files, _, _, _ = select_rand_models(MODELS_DIR, seed)
        #np.random.shuffle(selected_files)  # Si NO mezclaste durante el entrenamiento, no mezcles aquí

        # 3. Calcula features stacking: proba de cada modelo base
        test_preds  = pd.DataFrame()
        for idx, f in enumerate(selected_files):
            with open(f, "rb") as pf:
                model_info = pickle.load(pf)
            col_name = f"model_{idx}"
            test_preds[col_name] = predict_with_model(model_info, X_test)

        # Añade la columna objetivo (importante para calcular métricas en evaluación)
        test_preds[label] = y_test.values

        # 4. Guarda el DataFrame en la carpeta del ensamble correspondiente
        ens_dir = os.path.join(ENSEMBLES_DIR, f'ensemble_seed_{seed}')
        os.makedirs(ens_dir, exist_ok=True)
        test_preds.to_csv(os.path.join(ens_dir, 'test_preds.csv'), index=False)
        print(f"Guardado: {os.path.join(ens_dir, 'test_preds.csv')}")

def train_gluon_simulator(seed, MODELS_DIR, implementation_dir):
    selected_files, NN_sel, XGB_sel, SVM_sel = select_rand_models(MODELS_DIR, seed)

    # extraer seeds
    NN_seeds  = [extract_seed(f) for f in NN_sel]
    XGB_seeds = [extract_seed(f) for f in XGB_sel]
    SVM_seeds = [extract_seed(f) for f in SVM_sel]
    seeds_dict = {
        "NN": NN_seeds,     
        "XGB": XGB_seeds,   
        "SVM": SVM_seeds   
    }

    # guarda en la carpeta implementation el ensemble autogluon mejor entrenado en la ultima corrida de run.py
    best_autogluon_dir= os.path.join('classifier/modelos/Gluon', f"ensemble_seed_{seed}")
    preparar_implementation(best_autogluon_dir, seeds_dict, selected_files, seed, implementation_dir)

# main ------------------------------------------------------------------------
def main(train_n):
    # Leer el CSV
    csv_path = "database_creation/confirmed_umbrales_ciclones.csv"
    MODELS_DIR = 'classifier/modelos'
    ENSEMBLES_DIR = os.path.join(MODELS_DIR, "Gluon")
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    #genera las test_preds.csv para cada ensemble de gluon, con las probabilidades de cada modelo base (NN, SVM, XGB) como features
    generate_test_preds_for_gluon_ensembles(df, MODELS_DIR, ENSEMBLES_DIR, split_and_preprocess, predict_with_model, label='label', n_seeds=train_n)

    types = {"Gluon": ENSEMBLES_DIR}
    dataframes=[]
    best_seed = None

    #Esto solo se hace para evaluar cuál es el mejor modelo
    # 3) Evalúa cada tipo
    for model_type, path in types.items():
        if model_type == "Gluon":
            eval_fn = eval_dir_gluon
        y_tests, y_probas, seeds, roc_scores, pr_scores, best_pr, best_seed = eval_fn(
            path, df, label='label', n_seeds=train_n
        )
        print(f'Semilla: {best_seed} con el mejor AUC-PR={best_pr:.4f}')

    #Guardar en carpetas el mejor gluon encontrado
    implementation_dir = "implementation/trained_ensemble"
    train_gluon_simulator(best_seed, MODELS_DIR, implementation_dir)

if __name__ == "__main__":
    main(train_n)