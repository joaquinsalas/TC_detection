
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
from autogluon.tabular import TabularPredictor
import os
from classifier.train.Gluon.Gluon_30_train import predict_with_model
from classifier.train_common import split_and_preprocess, calcula_PR_ascendente
from classifier.evaluation.models_performance import generate_test_preds_for_gluon_ensembles, truncated_mean, nn_bundle_predict_proba, svm_bundle_predict_proba
import numpy as np
import re



def full_stats(data):
    return np.mean(data), np.std(data, ddof=1), len(data)

#modificado para que calcule las métricas de los valores entre los percentiles 5-95
def metricas(roc_scores, pr_scores, model_type):
    #truncado
    mean_roc, std_roc, n_roc = truncated_mean(roc_scores, 5, 95)
    mean_pr, std_pr, n_pr = truncated_mean(pr_scores, 5, 95)
    print(f"\n== Métricas para {model_type} ==")
    print(f"ROC AUC (media p5-p95): {mean_roc:.4f} ± {std_roc:.6f}  usando {n_roc} valores")
    print(f"PR  AUC (media p5-p95): {mean_pr:.4f} ± {std_pr:.6f}  usando {n_pr} valores")

    # --- COMPLETO (100%)
    mean_roc_f, std_roc_f, n_roc_f = full_stats(roc_scores)
    mean_pr_f, std_pr_f, n_pr_f = full_stats(pr_scores)
    print("\n--- Usando 100% de las muestras ---")
    print(f"ROC AUC: {mean_roc_f:.4f} ± {std_roc_f:.6f}  (n={n_roc_f})")
    print(f"PR  AUC: {mean_pr_f:.4f} ± {std_pr_f:.6f}  (n={n_pr_f})")

def paired_z_test(scores_A, scores_B, name_A, name_B):
    """
    Realiza un z-test pareado entre dos modelos usando sus scores por seed.
    """

    scores_A = np.array(scores_A)
    scores_B = np.array(scores_B)

    assert len(scores_A) == len(scores_B), "Ambos modelos deben tener el mismo número de seeds"

    # diferencias pareadas
    d = scores_A - scores_B

    mean_d = np.mean(d)
    std_d = np.std(d, ddof=1)
    n = len(d)

    # evitar división por cero
    if std_d == 0:
        print(f"\n⚠️ Desviación cero en diferencias entre {name_A} y {name_B}")
        return None

    z = mean_d / (std_d / np.sqrt(n))

    print(f"\n=== Paired z-test: {name_A} vs {name_B} ===")
    print(f"Mean difference: {mean_d:.6f}")
    print(f"Std difference:  {std_d:.6f}")
    print(f"z-score:         {z:.4f}")

    # interpretación con alpha = 0.05
    if abs(z) > 1.96:
        print("✅ Diferencia estadísticamente significativa (α = 0.05)")
    else:
        print("❌ No hay diferencia estadísticamente significativa (α = 0.05)")

    return z

def unpaired_z_test(scores_A, scores_B, name_A, name_B, n=30):
    """
    Z-test no pareado para diferencia de medias.
    Usa medias y desviaciones estándar de cada modelo.
    """

    scores_A = np.array(scores_A)
    scores_B = np.array(scores_B)

    # medias
    mu1 = np.mean(scores_A)
    mu2 = np.mean(scores_B)

    # desviaciones estándar muestrales
    sigma1 = np.std(scores_A, ddof=1)
    sigma2 = np.std(scores_B, ddof=1)

    # cálculo del z
    denom = np.sqrt((sigma1**2 / n) + (sigma2**2 / n))

    if denom == 0:
        print(f"\n⚠️ División por cero en {name_A} vs {name_B}")
        return None

    z = (mu1 - mu2) / denom

    print(f"\n=== Unpaired z-test: {name_A} vs {name_B} ===")
    print(f"Mean {name_A}: {mu1:.6f}")
    print(f"Mean {name_B}: {mu2:.6f}")
    print(f"Std  {name_A}: {sigma1:.6f}")
    print(f"Std  {name_B}: {sigma2:.6f}")
    print(f"z-score:       {z:.4f}")

    # interpretación (alpha = 0.05)
    if abs(z) > 1.96:
        print("✅ Diferencia estadísticamente significativa (α = 0.05)")
    else:
        print("❌ No hay diferencia estadísticamente significativa (α = 0.05)")

    return z

def compare_models(results, model_A, model_B, metric="roc", n=None):
    if n==None:
        return paired_z_test(
            results[model_A][metric],
            results[model_B][metric],
            model_A,
            model_B
        )
    else:
        return unpaired_z_test(
            results[model_A][metric],
            results[model_B][metric],
            model_A,
            model_B,
            n
        )

def eval_dir_gluon(base_dir, df, label, n_seeds=30):
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

    return y_tests, y_probas, seeds, roc_scores, pr_scores

def eval_dir(path, df, label):
    """Evalúa todos los PKL en 'path' y devuelve listas."""
    y_tests, y_probas, seeds = [], [], []
    roc_scores, pr_scores = [], []

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

    # return también por zona
    return y_tests, y_probas, seeds, roc_scores, pr_scores

# main ------------------------------------------------------------------------
def main():
    # Leer el CSV
    csv_path = "database_creation/confirmed_umbrales_ciclones.csv"
    base = 'classifier/modelos'
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    results = {}

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

    # 3) Evalúa cada tipo
    for model_type, path in types.items():
        if model_type == "Gluon":
            eval_fn = eval_dir_gluon
        else:
            eval_fn = eval_dir
        y_tests, y_probas, seeds, roc_scores, pr_scores = eval_fn(
            path, df, label='label'
        )
        metricas(roc_scores, pr_scores, model_type)
        
        # guardar resultados
        results[model_type] = {
            "roc": roc_scores,
            "pr": pr_scores
        }
    # =========================
    # Z-TESTS PAREADOS
    # =========================
    """print(f'paired z-test ROC: Gluon vs XGB: {compare_models(results, "Gluon", "XGB", "roc")}')
    print(f'paired z-test ROC: Gluon vs SVM: {compare_models(results, "Gluon", "SVM", "roc")}')
    print(f'paired z-test ROC: Gluon vs NN: {compare_models(results, "Gluon", "NN", "roc")}')

    print(f'paired z-test PR: Gluon vs XGB: {compare_models(results, "Gluon", "XGB", "pr")}')
    print(f'paired z-test PR: Gluon vs SVM: {compare_models(results, "Gluon", "SVM", "pr")}')
    print(f'paired z-test PR: Gluon vs NN: {compare_models(results, "Gluon", "NN", "pr")}')"""

    # =========================
    # Z-TESTS no PAREADOS
    # =========================
    print(f'unpaired z-test ROC: Gluon vs XGB: {compare_models(results, "Gluon", "XGB", "roc", 30)}')
    print(f'unpaired z-test ROC: Gluon vs SVM: {compare_models(results, "Gluon", "SVM", "roc", 30)}')
    print(f'nupaired z-test ROC: Gluon vs NN: {compare_models(results, "Gluon", "NN", "roc", 30)}')

    print(f'unpaired z-test PR: Gluon vs XGB: {compare_models(results, "Gluon", "XGB", "pr", 30)}')
    print(f'unpaired z-test PR: Gluon vs SVM: {compare_models(results, "Gluon", "SVM", "pr", 30)}')
    print(f'unpaired z-test PR: Gluon vs NN: {compare_models(results, "Gluon", "NN", "pr", 30)}')


if __name__ == "__main__":
    main()
