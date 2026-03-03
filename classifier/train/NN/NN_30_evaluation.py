# clasificador XGBoosting que identifica por el num de trayectorias y la dispersion cuando un ciclón es maduro o no
#con label confirmed
# En este archivo solo se hacen las evaluaciones y se generan métricas

#Con este script se hizo el 'best' que se usará en el ensamble ahora mismo, pero el mejor creo que será el que saldrá de los 'funciona'

"""
== Métricas para NN antes de poner el OnecyclLR==
Mean ROC AUC : 0.8864988772455089
Std  ROC AUC : 0.015386599576413761
Mean PR  AUC : 0.7303845784332847
Std  PR  AUC : 0.044609044661187
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import os
from NN_30_train import split_and_preprocess, calcula_PR_ascendente, FlexibleNN
import re
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns

plt.style.use("dark_background")
color_graph = 'black'

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

def eval_dir(path, df, label,zona_split_col=None):
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

        
        # Si es un bundle NN (dict), reconstruimos; si no, asumimos sklearn (XGB Pipeline)
        if isinstance(clf, dict) and "model_state" in clf and "scaler" in clf:
            # Asegúrate de pasar las mismas 3 features y en el mismo orden
            # ["n_trayectorias_best_cluster", "dispersión_km_best_cluster", "horas_diff_estimadas"]
            y_proba = nn_bundle_predict_proba(clf, X_test)

        # Si es un bundle SVM (dict con 'model'), usamos scikit-learn
        elif isinstance(clf, dict) and "model" in clf and "scaler" in clf:
            y_proba = svm_bundle_predict_proba(clf, X_test)
        else:
            # XGB / sklearn pipeline
            y_proba = clf.predict_proba(X_test)[:, 1]
        #y_proba = clf.predict_proba(X_test)[:,1]
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

def build_results_df(model_type, zone_roc_scores, zone_pr_scores):
    rows = []
    for zona in ['atlantico', 'pacifico']:
        # ROC
        for val in zone_roc_scores[zona]:
            rows.append({'modelo': model_type, 'zona': zona, 'metrica': 'ROC AUC', 'valor': val})
        # PR
        for val in zone_pr_scores[zona]:
            rows.append({'modelo': model_type, 'zona': zona, 'metrica': 'PR AUC', 'valor': val})
    return pd.DataFrame(rows)

def bloxplot_metricas(df_box):
    # Boxplot separado por métrica (ROC/PR) y zona, y color para el modelo:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, metrica in enumerate(['ROC AUC', 'PR AUC']):
        sns.boxplot(
            data=df_box[df_box['metrica'] == metrica],
            x="zona",
            y="valor",
            hue="modelo",
            ax=axes[i],
            #showmeans=True,
            palette="bright",
        )
        # Opcional: agrega los puntos individuales
        sns.stripplot(
            data=df_box[df_box['metrica'] == metrica],
            x="zona",
            y="valor",
            hue="modelo",
            ax=axes[i],
            dodge=True,
            color='white',
            alpha=0.3,
            marker='o',
            palette="bright"
        )
        axes[i].set_title(metrica)
        axes[i].set_ylabel("Value")
        axes[i].set_xlabel("Zone")
        axes[i].set_xticklabels(['Atlantic', 'Pacific'])
        axes[i].grid(True, linestyle='--', color='gray', alpha=0.5)
        # Quitar leyenda duplicada
        handles, labels = axes[i].get_legend_handles_labels()
        if i == 0:
            axes[i].legend(handles[:3], labels[:3], title="Model")
        else:
            axes[i].get_legend().remove()

    plt.suptitle("Distribution of ROC AUC and PR AUC by Zone and Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/boxplot_auc_por_zona_y_modelo.png", dpi=300, bbox_inches='tight')

# main ------------------------------------------------------------------------
def main():
    # Leer el CSV
    csv_path = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv"
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    base = '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos'
    types = {"XGB": os.path.join(base, "XGB"),
             "NN" : os.path.join(base, "NN"),
             "SVM": os.path.join(base, "SVM")
             }
    zona_split_col = 'zona'
    dataframes=[]



    # 3) Evalúa cada tipo
    for model_type, path in types.items():
        y_tests, y_probas, seeds, roc_scores, pr_scores, zone_roc_scores, zone_pr_scores = eval_dir(
            path, df, label='label', zona_split_col=zona_split_col
        )
        grafica_roc_pr(y_tests, y_probas, seeds, model_type)
        metricas(roc_scores, pr_scores, model_type)

        dataframe = build_results_df(model_type, zone_roc_scores, zone_pr_scores)
        dataframes.append(dataframe)

        # --- NUEVO: imprimir métricas por zona ---
        for zona in ['atlantico', 'pacifico']:
            print(f"\n== Métricas para {model_type} en zona {zona} ==")
            mean_roc, std_roc, n_roc = truncated_mean(zone_roc_scores[zona], 5, 95)
            mean_pr, std_pr, n_pr = truncated_mean(zone_pr_scores[zona], 5, 95)
            print(f"ROC AUC (media p5-p95): {mean_roc:.4f} ± {std_roc:.6f}  usando {n_roc} valores")
            print(f"PR  AUC (media p5-p95): {mean_pr:.4f} ± {std_pr:.6f}  usando {n_pr} valores")
        
    #Concatena los DataFrames:
    df_box = pd.concat([dataframes[0], dataframes[1], dataframes[2]], ignore_index=True)
    bloxplot_metricas(df_box)


        


if __name__ == "__main__":
    main()