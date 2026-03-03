
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
from SVM_30_train import split_and_preprocess, calcula_PR_ascendente
import re
import torch
import torch.nn as nn
import numpy as np


#plt.style.use("dark_background")
color_graph = 'white'

def svm_bundle_predict_proba(svm_bundle, X_test):
    scaler = svm_bundle["scaler"]
    model = svm_bundle["model"]

    # Asegura numpy 2D
    X_np = X_test.values if hasattr(X_test, "values") else X_test
    Xs = scaler.transform(X_np)
    probas = model.predict_proba(Xs)[:, 1]
    return probas



#definición de métodos
def grafica_roc_pr(y_tests, y_probas, seeds, model_type):
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
    ax.set_title(f"{model_type} - ROC Curve")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/figs/roc_curve_{model_type}_30_{color_graph}.png", dpi=300, bbox_inches="tight", pad_inches=0.2)

    # === CURVA PRECISION–RECALL (escalera) ===
    # 1) DataFrame y orden descendente por recall
    fig, ax = plt.subplots(figsize=(8, 6))
    for y_true, y_proba, seed in zip(y_tests, y_probas, seeds):
        #calcula el AUC_PR escalonado para mejor
        auc_pr_f, recall_f, precision_f = calcula_PR_ascendente(y_true, y_proba)
        ax.plot(recall_f, precision_f)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_type} - Precision–Recall Curve")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/figs/pr_curve_{model_type}_30_{color_graph}.png", dpi=300, bbox_inches="tight", pad_inches=0.2)

def metricas(roc_scores, pr_scores, model_type):
    print(f"\n== Métricas para {model_type} ==")
    print("Mean ROC AUC :", np.mean(roc_scores))
    print("Std  ROC AUC :", np.std(roc_scores))
    print("Mean PR  AUC :", np.mean(pr_scores))
    print("Std  PR  AUC :", np.std(pr_scores))

def eval_dir(path, df, label, zona_split_col=None):
    """Evalúa todos los PKL en 'path' y devuelve listas."""
    y_tests, y_probas, seeds = [], [], []
    roc_scores, pr_scores = [], []
    # Para zonas
    zones = ['atlantico', 'pacifico']
    zone_roc_scores = {z: [] for z in zones}
    zone_pr_scores = {z: [] for z in zones}

    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".pkl"):
            continue
        m = re.search(r"seed_(\d+)\.pkl$", fname)
        if not m:
            continue
        seed = int(m.group(1))

        _, X_test, _, y_test = split_and_preprocess(
            df, date_col='fecha_prediccion',
            train_start='2023-01-01', train_end='2024-12-31',
            label=label, seed=seed
        )

        # Cargar zona para test
        zonas_test = df.loc[X_test.index, zona_split_col] if zona_split_col else None

        # carga modelo
        with open(os.path.join(path, fname), "rb") as f:
            clf = pickle.load(f)

        
        # Si es un bundle SVM (dict con 'model'), usamos scikit-learn
        if isinstance(clf, dict) and "model" in clf and "scaler" in clf:
            y_proba = svm_bundle_predict_proba(clf, X_test)
        else:
            # XGB / sklearn pipeline
            y_proba = clf.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
        pr, _, _ = calcula_PR_ascendente(y_test, y_proba)

        y_tests.append(y_test)
        y_probas.append(y_proba)
        seeds.append(seed)
        roc_scores.append(roc)
        pr_scores.append(pr)

    # --- NUEVO: métricas por zona ---
        if zona_split_col is not None and zonas_test is not None:
            for zona in zones:
                mask = (zonas_test == zona)
                if mask.sum() > 0:  # hay datos para esa zona
                    roc_z = roc_auc_score(y_test[mask], y_proba[mask])
                    pr_z, _, _ = calcula_PR_ascendente(y_test[mask], y_proba[mask])
                    zone_roc_scores[zona].append(roc_z)
                    zone_pr_scores[zona].append(pr_z)
                else:
                    zone_roc_scores[zona].append(np.nan)
                    zone_pr_scores[zona].append(np.nan)
    # return también por zona
    return y_tests, y_probas, seeds, roc_scores, pr_scores, zone_roc_scores, zone_pr_scores


# main ------------------------------------------------------------------------
def main():
    # Leer el CSV
    csv_path = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv"
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    base = '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos'
    types = {#"XGB": os.path.join(base, "XGB_best"),
            #"NN" : os.path.join(base, "NN_best"),
            "SVM": os.path.join(base, "SVM_best")
    }
    zona_split_col = 'zona'

    # 3) Evalúa cada tipo
    for model_type, path in types.items():
        y_tests, y_probas, seeds, roc_scores, pr_scores, zone_roc_scores, zone_pr_scores = eval_dir(
            path, df, label='label', zona_split_col=zona_split_col
        )
        grafica_roc_pr(y_tests, y_probas, seeds, model_type)
        metricas(roc_scores, pr_scores, model_type)

        # --- NUEVO: imprimir métricas por zona ---
        for zona in ['atlantico', 'pacifico']:
            print(f"\n== Métricas para {model_type} en zona {zona} ==")
            print("Mean ROC AUC :", np.nanmean(zone_roc_scores[zona]))
            print("Std  ROC AUC :", np.nanstd(zone_roc_scores[zona]))
            print("Mean PR  AUC :", np.nanmean(zone_pr_scores[zona]))
            print("Std  PR  AUC :", np.nanstd(zone_pr_scores[zona]))


if __name__ == "__main__":
    main()