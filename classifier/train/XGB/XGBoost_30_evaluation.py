# clasificador XGBoosting que identifica por el num de trayectorias y la dispersion cuando un ciclón es maduro o no
#con label confirmed
# En este archivo solo se hacen las evaluaciones y se generan métricas

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import os
from XGBoost_30_train import split_and_preprocess, calcula_PR_ascendente
import re
from sklearn.metrics import precision_recall_curve

#plt.style.use("dark_background")
color_graph = 'white'

#definición de métodos
def grafica_roc_pr(y_tests, y_probas, seeds):
    # === CURVA ROC ===
    fig, ax = plt.subplots(figsize=(8, 6))
    for y_true, y_proba, seed in zip(y_tests, y_probas, seeds):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr)
    # Diagonal punteada blanca desde (0,0) hasta (1,1)
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label="Random (50%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("XGB - ROC Curve")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/figs/roc_curve_XGB_30_{color_graph}.png", dpi=300, bbox_inches="tight", pad_inches=0.2)

    # === CURVA PRECISION–RECALL (escalera) ===
    # 1) DataFrame y orden descendente por recall
    fig, ax = plt.subplots(figsize=(8, 6))
    for y_true, y_proba, seed in zip(y_tests, y_probas, seeds):
        #calcula el AUC_PR escalonado para mejor
        auc_pr_f, recall_f, precision_f = calcula_PR_ascendente(y_true, y_proba) # este es el bueno
        precision_f, recall_f, thresholds = precision_recall_curve(y_true, y_proba)
        ax.plot(recall_f, precision_f)
        if seed ==4: #solo muestra los umbrales usados para la mejor semilla
            for i in range(0, len(thresholds), 2):  # cada 10 puntos
                plt.text(recall_f[i], precision_f[i], f"{thresholds[i]:.2f}", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("XGB - Precision–Recall Curve")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/figs/pr_curve_XGB_30_{color_graph}.png", dpi=300, bbox_inches="tight", pad_inches=0.2)

def metricas(roc_scores, pr_scores):
    print("Media ROC:", np.mean(roc_scores))
    print("Desviación estándar ROC:", np.std(roc_scores))
    print("Media PR:", np.mean(pr_scores))
    print("Desviación estándar PR:", np.std(pr_scores))

# main ------------------------------------------------------------------------
def main():
    # Leer el CSV
    model_dir = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos/XGB_best"
    csv_path = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv"
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])
    y_tests, y_probas, seeds = [], [], []
    roc_scores, pr_scores = [], []


    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".pkl"):
            continue
        # Extraer semilla del nombre
        m = re.search(r"seed_(\d+)\.pkl$", fname)
        if not m:
            print(f"Warning: no pude extraer seed de '{fname}', lo salto.")
            continue
        seed = int(m.group(1))

    
        # Train-test split
        _, X_test, _, y_test = split_and_preprocess(
            df,
            date_col='fecha_prediccion',
            train_start='2023-01-01',
            train_end  ='2024-12-31',
            label='label',
            seed = seed
        )

        # 4.2) Cargar el clasificador guardado
        model_path = os.path.join(model_dir, fname)
        with open(model_path, "rb") as f:
            clf = pickle.load(f)

        # 2.3) Predicción y métricas
        y_proba = (clf.predict_proba(X_test)[:, 1]>0.45).astype(float)
        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr, _, _ = calcula_PR_ascendente(y_test, y_proba)
        roc_scores.append(auc_roc)
        pr_scores.append(auc_pr)
        #guardar los resutlados paa después graficarlos
        y_tests.append(y_test)
        y_probas.append(y_proba)
        seeds.append(seed)

    #grafica    
    grafica_roc_pr(y_tests, y_probas, seeds)

    #guarda los datos necesarios para generar las graficas en blanco y negro 
    # === Guardar curvas ROC y PR por seed en CSV (60 archivos en total) ===
    roc_pr_dir = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/XGB/rocs_prs"
    for seed, y_true, y_proba in zip(seeds, y_tests, y_probas):
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)    
        pd.DataFrame({"fpr": fpr, "tpr": tpr})\
        .to_csv(f"{roc_pr_dir}/roc_seed_{seed}.csv", index=False)
        # PR (versión escalonada que ya usas en las gráficas)
        auc_pr_f, recall_f, precision_f = calcula_PR_ascendente(y_true, y_proba)
        pd.DataFrame({"recall": recall_f, "precision": precision_f})\
        .to_csv(f"{roc_pr_dir}/pr_seed_{seed}.csv", index=False)
    
    #obtiene la media y la desvaición estandar de las métricas de los clasificadores:
    metricas(roc_scores, pr_scores)


if __name__ == "__main__":
    main()