#aqui se extrae la info del mejor modelo autogluon
#primero se deben evaluar todos para extraer quel que haya obtenido un AUC_PR más alto y a él se le sacarán las características

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import os
from Gluon_30_train import split_and_preprocess, calcula_PR_ascendente, FlexibleNN, predict_with_model
import re
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from autogluon.tabular import TabularPredictor
import glob


plt.style.use("dark_background")
color_graph = 'black'


#definición de métodos
def truncated_mean(data, pmin=10, pmax=90):
    lower = np.percentile(data, pmin)
    upper = np.percentile(data, pmax)
    filtered = [x for x in data if (x >= lower and x <= upper)]
    return np.mean(filtered), np.std(filtered), len(filtered)

#modificado para que calcule las métricas de los valores entre los percentiles 5-95
def metricas(roc_scores, pr_scores, model_type):
    mean_roc, std_roc, n_roc = truncated_mean(roc_scores, 5, 95)
    mean_pr, std_pr, n_pr = truncated_mean(pr_scores, 5, 95)

    print(f"\n== Métricas para {model_type} ==")
    print(f"ROC AUC (media p5-p95): {mean_roc:.4f} ± {std_roc:.6f}  usando {n_roc} valores")
    print(f"PR  AUC (media p5-p95): {mean_pr:.4f} ± {std_pr:.6f}  usando {n_pr} valores")

def generate_test_preds_for_gluon_ensembles(
    df, MODELS_DIR, ENSEMBLES_DIR, split_and_preprocess, predict_with_model,
    label='label', n_seeds=30):

    for seed in range(n_seeds):
        print(f"\n=== SEMILLA {seed} ===")
        # 1. Split como en entrenamiento
        _, X_test, _, y_test = split_and_preprocess(
            df, 'fecha_prediccion', '2023-01-01', '2024-12-31', label, seed
        )

        # 2. Selecciona 3 modelos al azar de cada tipo con la misma semilla
        NN_files  = glob.glob(os.path.join(f'{MODELS_DIR}/NN',  "NN_classifier_seed_*.pkl"))
        XGB_files = glob.glob(os.path.join(f'{MODELS_DIR}/XGB', "XGB_classiffier_seed_*.pkl"))
        SVM_files = glob.glob(os.path.join(f'{MODELS_DIR}/SVM', "SVM_classifier_seed_*.pkl"))
        NN_sel  = np.random.RandomState(seed).choice(NN_files,  3, replace=False)
        XGB_sel = np.random.RandomState(seed).choice(XGB_files, 3, replace=False)
        SVM_sel = np.random.RandomState(seed).choice(SVM_files, 3, replace=False)
        selected_files = list(NN_sel) + list(XGB_sel) + list(SVM_sel)
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

def eval_dir_gluon(base_dir, df, label, zona_split_col=None, n_seeds=30):
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

def leaderboard_to_latex(df, 
                         caption="Models in the AutoGluon Ensemble", 
                         label="tab:autogluon_ensemble"):
    """
    Recibe el DataFrame devuelto por predictor.leaderboard(silent=False)
    y devuelve código LaTeX para una tabla con las columnas:
    model, stack_level, fit_order.
    Se escapan los '_' para evitar errores de compilación en LaTeX.
    """

    # Columnas relevantes
    cols = ["model", "stack_level", "fit_order", "score_val"]
    df_sub = df[cols].copy()

    # Construir filas LaTeX
    latex_rows = ""
    for _, row in df_sub.iterrows():
        model_name = str(row["model"]).replace("_", "\\_")  # <--- ESCAPE AQUÍ
        stack_level = row["stack_level"]
        fit_order = row["fit_order"]
        score_val = row["score_val"]
        latex_rows += f"{model_name} & {stack_level} & {fit_order} & {round(score_val, 4)}\\\\\n"

    # Tabla LaTeX completa
    latex_table = f"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\\begin{{table}}[h!]
\\centering
\\caption{{{caption}}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Model}} & \\textbf{{Stack level}} & \\textbf{{Fit order}} & \\textbf{{Score val}}\\\\
\\midrule
{latex_rows}\\bottomrule
\\end{{tabular}}
\\label{{{label}}}
\\end{{table}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
    return latex_table

def predictor_info(base_dir, seed):
    # Carga el predictor AutoGluon
    ens_dir = os.path.join(base_dir, f'ensemble_seed_{seed}')
    predictor = TabularPredictor.load(ens_dir)
    ensemble_model = predictor._trainer.load_model("WeightedEnsemble_L3") #cargar el ensamble final

    #obtener los pesos del ensamble final
    weights = ensemble_model._get_model_weights()
    models = ensemble_model.base_model_names

    for m, w in zip(models, weights):
        print(m, w)

    lb = predictor.leaderboard(silent=False)
    print(leaderboard_to_latex(lb))







# main ------------------------------------------------------------------------
def main():
    # Leer el CSV
    csv_path = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv"
    base = '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos'
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    types = {"Gluon": os.path.join(base, "Gluon")}
    zona_split_col = 'zona'
    dataframes=[]

    #Esto solo se hace para evaluar cuál es el mejor modelo
    """# 3) Evalúa cada tipo
    for model_type, path in types.items():
        if model_type == "Gluon":
            eval_fn = eval_dir_gluon
        y_tests, y_probas, seeds, roc_scores, pr_scores, best_pr, best_seed = eval_fn(
            path, df, label='label', zona_split_col=zona_split_col
        )
        metricas(roc_scores, pr_scores, model_type)

        print(f'Semilla: {best_seed} con el mejor AUC-PR={best_pr:.4f}') # fue la 2"""

    #ahora solo se va a dedicar a scar información de la semilla 2
    seed = 2
    good_dir = '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos/Gluon'
    predictor_info(good_dir, seed)


if __name__ == "__main__":
    main()