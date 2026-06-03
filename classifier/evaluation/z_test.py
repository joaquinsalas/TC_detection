import numpy as np

"""# Número de corridas / semillas
n = 26
# Resultados de la tabla
# Formato: "clasificador": {"metric": (mean, std)}
results = {
    "XGB": {
        "ROC-AUC": (0.906, 0.006),
        "PR-AUC":  (0.856, 0.008),
    },
    "NN": {
        "ROC-AUC": (0.912, 0.005),
        "PR-AUC":  (0.853, 0.011),
    },
    "SVM": {
        "ROC-AUC": (0.902, 0.01),
        "PR-AUC":  (0.854, 0.012),
    },
    "Gluon": {
        "ROC-AUC": (0.904, 0.007),
        "PR-AUC":  (0.858, 0.007),
    }
}
"""

def compute_z(mean_1, std_1, n_1, mean_2, std_2, n_2):
    z = (mean_1 - mean_2) / np.sqrt((std_1**2 / n_1) + (std_2**2 / n_2))
    return z

def main(results):
    reference_model = "Gluon"
    for metric in ["ROC-AUC", "PR-AUC"]:
        print(f"\nZ-values for {metric}")
        print("-" * 35)

        mean_ref, std_ref, n_ref = results[reference_model][metric]

        for classifier in results:
            if classifier == reference_model:
                continue

            mean_cls, std_cls, n_cls = results[classifier][metric]

            z = compute_z(
                mean_ref, std_ref, n_ref,
                mean_cls, std_cls, n_cls
            )
            # interpretación (alpha = 0.05)
            if abs(z) > 1.96:
                message="✅ Diferencia estadísticamente significativa (α = 0.05)"
            else:
                message= "❌ No hay diferencia estadísticamente significativa (α = 0.05)"

            print(
                f"{reference_model} vs {classifier}: "
                f"z = {z:.2f}, |z| = {abs(z):.2f} "
                f"{message}"
            )

if __name__ == "__main__":
    main(results)
