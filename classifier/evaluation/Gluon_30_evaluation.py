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
from Gluon_30_train import split_and_preprocess, calcula_PR_ascendente, FlexibleNN, predict_with_model
import re
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from autogluon.tabular import TabularPredictor
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import precision_recall_curve

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

#el otro
def grafica_roc_pr_area(y_tests, y_probas, seeds, model_type):
    # ----------- CURVA ROC --------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Grilla uniforme de FPR para todas las curvas
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for y_true, y_proba, seed in zip(y_tests, y_probas, seeds):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        # Interpola cada curva en la grilla común
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    tpr_min = np.min(tprs, axis=0)
    tpr_max = np.max(tprs, axis=0)
    tpr_mean = np.mean(tprs, axis=0)
    tpr_p5 = np.percentile(tprs, 5, axis=0)
    tpr_p95 = np.percentile(tprs, 95, axis=0)

    # Área entre percentiles (más robusto que min-max)
    ax.fill_between(mean_fpr, tpr_p5, tpr_p95, color="dodgerblue", alpha=0.3, label="5–95 percentile")
    # (Opcional) Curva media
    ax.plot(mean_fpr, tpr_mean, color="dodgerblue", lw=2, label="ROC Mean")

    # Línea aleatoria
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
    plt.close(fig)

    # ----------- CURVA PR --------------
    fig, ax = plt.subplots(figsize=(8, 6))

    mean_recall = np.linspace(0, 1, 100)
    precisions = []

    for y_true, y_proba, seed in zip(y_tests, y_probas, seeds):
        auc_pr_f, recall_f, precision_f = calcula_PR_ascendente(y_true, y_proba)
        # Interpola precision en grilla uniforme de recall
        precision_interp = np.interp(mean_recall, recall_f[::-1], precision_f[::-1])
        precisions.append(precision_interp)

    precisions = np.array(precisions)
    prec_min = np.min(precisions, axis=0)
    prec_max = np.max(precisions, axis=0)
    prec_mean = np.mean(precisions, axis=0)
    prec_p5 = np.percentile(precisions, 5, axis=0)
    prec_p95 = np.percentile(precisions, 95, axis=0)

    # Banda entre percentiles
    ax.fill_between(mean_recall, prec_p5, prec_p95, color="green", alpha=0.3, label="5–95 percentile") #color="gold"
    ax.plot(mean_recall, prec_mean, color="green", lw=2, label="PR Mean") #color="gold"

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_type} - Precision–Recall Curve")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/figs/pr_curve_{model_type}_30_{color_graph}.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


#el bueno
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

#ubica el mejor threshold
def thresholds_PR(y_tests, y_probas, seeds, model_type):    
    # === CURVA PRECISION–RECALL (escalera) ===
    # 1) DataFrame y orden descendente por recall
    _, ax = plt.subplots(figsize=(8, 6))
    for y_true, y_proba, seed in zip(y_tests, y_probas, seeds):
        precision_f, recall_f, thresholds = precision_recall_curve(y_true, y_proba)
        ax.plot(recall_f, precision_f)
        #solo muestra los umbrales usados para la mejor semilla
        if seed ==2: #solo muestra los umbrales usados para la mejor semilla
            for i in range(0, len(thresholds), 2):  # cada 10 puntos
                plt.text(recall_f[i], precision_f[i], f"{thresholds[i]:.2f}", fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_type} - Precision–Recall Curve")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/figs/pr_curve_{model_type}_{seed}_{color_graph}_thresholds.png", dpi=300, bbox_inches="tight", pad_inches=0.2)

def truncated_mean(data, pmin=5, pmax=95):
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
        X_train, X_test, y_train, y_test = split_and_preprocess(
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
    y_tests, y_probas, seeds = [], [], []
    roc_scores, pr_scores = [], []
    zones = ['atlantico', 'pacifico']
    zone_roc_scores = {z: [] for z in zones}
    zone_pr_scores = {z: [] for z in zones}

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

        # Métricas por zona (si aplica)
        if zona_split_col is not None:
            # Recupera los índices originales
            _, X_test_df, _, _ = split_and_preprocess(
                df, 'fecha_prediccion', '2023-01-01', '2024-12-31', label, seed
            )
            zonas_test = df.loc[X_test_df.index, zona_split_col]
            for zona in zones:
                mask = (zonas_test == zona).values
                if mask.sum() > 0:
                    roc_z = roc_auc_score(y_test[mask], y_proba[mask])
                    pr_z, _, _ = calcula_PR_ascendente(y_test[mask], y_proba[mask])
                    zone_roc_scores[zona].append(roc_z)
                    zone_pr_scores[zona].append(pr_z)
                else:
                    zone_roc_scores[zona].append(np.nan)
                    zone_pr_scores[zona].append(np.nan)

    return y_tests, y_probas, seeds, roc_scores, pr_scores, zone_roc_scores, zone_pr_scores

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
        axes[i].set_xlim(-0.5, 1.5)  # Para evitar que se encimen las cajas
        axes[i].set_ylim(0, 1)
        # Boxplot principal
        sns.boxplot(
            data=df_box[df_box['metrica'] == metrica],
            x="zona",
            y="valor",
            hue="modelo",
            ax=axes[i],
            palette="bright",
        )
        # Puntos individuales
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
            axes[i].legend(handles[:4], labels[:4], title="Model")
        else:
            axes[i].get_legend().remove()

        # --- Agregar zoom (inset axes) ---
        # Definir los límites del zoom para cada métrica
        if metrica == 'ROC AUC':
            y1, y2 = 0.95, 1.0
        else:  # 'PR AUC'
            y1, y2 = 0.75, 1.0
        # Crear el recuadro en la esquina inferior izquierda
        axins = inset_axes(axes[i], width="75%", height="75%", loc='lower right', 
                           bbox_to_anchor=(0.07, 0.07, 0.9, 0.9), bbox_transform=axes[i].transAxes)
        # Repetir boxplot y puntos en el recuadro
        sns.boxplot(
            data=df_box[df_box['metrica'] == metrica],
            x="zona",
            y="valor",
            hue="modelo",
            ax=axins,
            palette="bright",
            linewidth=1,
            fliersize=0
        )
        sns.stripplot(
            data=df_box[df_box['metrica'] == metrica],
            x="zona",
            y="valor",
            hue="modelo",
            ax=axins,
            dodge=True,
            color='white',
            alpha=0.3,
            marker='o',
            palette="bright",
            size=3
        )
        axins.set_ylim(y1, y2)
        axins.set_xticklabels([])
        axins.set_xlabel("")
        axins.set_ylabel("")
        axins.set_yticks([y1, y2])
        axins.tick_params(axis='both', which='major', labelsize=7)
        axins.grid(True, linestyle='--', color='gray', alpha=0.5)
        # Sin leyenda en los insets
        axins.get_legend().remove() if axins.get_legend() else None

    plt.suptitle("Distribution of ROC AUC and PR AUC by Zone and Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/boxplot_auc_por_zona_y_modelo.png", dpi=300, bbox_inches='tight')

# main ------------------------------------------------------------------------
def main():
    # Leer el CSV
    csv_path = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv"
    base = '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos'
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    #generar el test_preds para los ensambles Gluon
    # LLAMA LA FUNCIÓN (ajusta nombre y firma de tus helpers si difieren)
    generate_test_preds_for_gluon_ensembles(
        df=df,
        MODELS_DIR=base,
        ENSEMBLES_DIR=os.path.join(base, "Gluon"),
        split_and_preprocess=split_and_preprocess,
        predict_with_model=predict_with_model,
        label='label',
        n_seeds=30
    )

    
    types = {"XGB": os.path.join(base, "XGB"),
             "NN" : os.path.join(base, "NN"),
             "SVM": os.path.join(base, "SVM"),
             "Gluon": os.path.join(base, "Gluon")
             }
    zona_split_col = 'zona'
    dataframes=[]



    # 3) Evalúa cada tipo
    for model_type, path in types.items():
        if model_type == "Gluon":
            eval_fn = eval_dir_gluon
        else:
            eval_fn = eval_dir
        y_tests, y_probas, seeds, roc_scores, pr_scores, zone_roc_scores, zone_pr_scores = eval_fn(
            path, df, label='label', zona_split_col=zona_split_col
        )
        grafica_roc_pr_area(y_tests, y_probas, seeds, model_type)
        thresholds_PR(y_tests, y_probas, seeds, model_type) #temporal
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
    df_box = pd.concat([dataframes[0], dataframes[1], dataframes[2], dataframes[3]], ignore_index=True)
    bloxplot_metricas(df_box)


if __name__ == "__main__":
    main()