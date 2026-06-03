#hace más evaluaciones del ensemble ganador AutoGluon

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from common import ClasificadorHurakan, load_base_model_seeds
from classifier.train_common import split_and_preprocess, preprocess

def get_stats(df):
    n_total = len(df)
    n_class_0 = (df['label'] == 0).sum()
    n_class_1 = (df['label'] == 1).sum()
    n_unique_cyclones = df['SID'].nunique()

    return {
        "class_0": n_class_0,
        "class_1": n_class_1,
        "total": n_total,
        "unique_SID": n_unique_cyclones
    }

def database_summary(confirmed_umbrales_path):
# Leer CSV
    df = pd.read_csv(confirmed_umbrales_path)
    seed = 42

    # Split train/test
    df['fecha_prediccion'] = pd.to_datetime(df['fecha_prediccion'])
    mask_train = df['fecha_prediccion'].between("2023-01-01", "2024-12-31")
    df_train = df[mask_train].copy()
    df_test  = df[~mask_train].copy()

    # Train balanced
    _, y_train_bal = preprocess(df_train, 'label', 'train', seed)

    # reconstruir df balanceado usando índices
    df_train_bal = df_train.loc[y_train_bal.index]

    # Calcular estadísticas
    stats = {
        "df_completo": get_stats(df),
        "train (2023-2024)": get_stats(df_train),
        "test (2025)": get_stats(df_test),
        "train_balanced": get_stats(df_train_bal)
    }

    # Convertir a tabla
    stats_df = pd.DataFrame(stats).T
    stats_df = stats_df[["class_0", "class_1", "total", "unique_SID"]]
    print("\nResumen del dataset:\n")
    print(stats_df)

def graph_aciertos_hora(df_confirmed_umbrales, horas="horas_diff_estimadas"):
    # 1. Crear columna de acierto (1 = correcto, 0 = incorrecto)
    df_confirmed_umbrales["acierto"] = (
        df_confirmed_umbrales["label"] == df_confirmed_umbrales["predicted_label"]
    ).astype(int)

    # 2. Agrupar por horas y calcular porcentaje de acierto
    accuracy_por_hora = (
        df_confirmed_umbrales
        .groupby(horas)["acierto"]
        .mean() * 100
    )

    # 3. Graficar
    plt.figure(figsize=(8, 5))
    accuracy_por_hora.plot(marker="o", color = 'green')
    plt.xlabel('Lead time (hours)') #horas
    plt.ylabel("Accuracy (%)")
    plt.title(f"Percentage of accuracy in each lead time")
    plt.grid(True)
    plt.tight_layout()
    # Forzar límites del eje Y
    plt.ylim(0, 105)

    # Ticks solo cada 20
    plt.yticks(range(0, 101, 20))
    #plt.xticks(range(-360, 0, 12), rotation=35)
    plt.savefig("classifier/fig/aciertos_hora.png", dpi=300)
    plt.close()

def graph_registros(df_confirmed_umbrales, horas="horas_diff_reales"):
    # Separar por clase
    df_0 = df_confirmed_umbrales[df_confirmed_umbrales["label"] == 0]
    df_1 = df_confirmed_umbrales[df_confirmed_umbrales["label"] == 1]

    # Definir bins (uno por cada hora entera)
    bins = sorted(df_confirmed_umbrales[horas].unique())

    plt.figure(figsize=(9, 5))

    plt.hist(
        [df_0[horas], df_1[horas]],
        bins=bins,
        stacked=True,
        color=["red", "green"],
        label=["Label = 0", "Label = 1"]
    )

    plt.xlabel(f'{horas}')
    plt.ylabel("Cantidad de registros")
    plt.title(f"Distribución de registros por {horas}")
    plt.legend()
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig("classifier/fig/registros_hora.png", dpi=300)
    plt.close()

    conteos = (
        df_confirmed_umbrales
        .groupby([horas, "label"])
        .size()
        .unstack(fill_value=0)
    )

    #print(conteos)


#dada la matriz de confusión, queremos disminuir los Fn, es decir, maximizar recall
def matriz(df_confirmed_umbrales, lead_time=None):
    title = "Confusion matrix"
    if lead_time is not None:
        df_confirmed_umbrales = df_confirmed_umbrales[df_confirmed_umbrales["horas_diff_estimadas"] == lead_time]
        title = f"Confusion matrix on lead time = {lead_time} hours"

    # Extraer etiquetas reales y predichas
    y_true = df_confirmed_umbrales["label"]
    y_pred = df_confirmed_umbrales["predicted_label"]

    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    # Graficar
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1]
    )
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1]
    )
    disp.plot(
        ax=ax,
        values_format="d",
        cmap="Blues",
        im_kw={"vmin": 0, "vmax": 220}
    )
    # Cambiar color dinámico del texto
    threshold = 100
    for text in disp.text_.ravel():
        value = int(text.get_text())
        if value < threshold:
            text.set_color("black")
        else:
            text.set_color("white")

    plt.title(title)
    #plt.tight_layout()
    plt.savefig("classifier/fig/confussion_matrix.png", dpi=300)
    plt.close()

    #sacar las métricas
    TN, FP, FN, TP = cm.ravel()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    #far = FP / (TN + FP) 
    far = FP / (TP + FP) 
    return precision, recall, accuracy, far

def main(color_graph, lead_time = None):
    if color_graph == 'black':
        plt.style.use("dark_background")

    trained_ensemble_dir = 'implementation/trained_ensemble'
    models_dir = os.path.join(trained_ensemble_dir, 'modelos_ensamble')
    json_dir = os.path.join(trained_ensemble_dir, 'base_models_seeds.json')
    confirmed_umbrales_path = 'database_creation/confirmed_umbrales_ciclones.csv'

    #contabiliza los registros en los sets de train, test, tran_balanceada, y la base de datos entera
    database_summary(confirmed_umbrales_path)

    #usa el mejor modelo entrenado
    folders = [f for f in os.listdir(trained_ensemble_dir) if f.startswith("ensemble_seed_")]
    ensemble_dir = os.path.join(trained_ensemble_dir, folders[0])
    
    df_confirmed_umbrales = pd.read_csv(
        confirmed_umbrales_path,
        usecols=["n_trayectorias_best_cluster","dispersión_km_best_cluster","horas_diff_estimadas","horas_diff_reales", "label", "NAME"],
    )

    # Es una aplicación de ejemplo con un solo cluster
    cluster_features = df_confirmed_umbrales[["n_trayectorias_best_cluster","dispersión_km_best_cluster","horas_diff_estimadas"]]

    #crea el objeto
    NN_seeds, XGB_seeds, SVM_seeds = load_base_model_seeds(json_dir)
    clasificador = ClasificadorHurakan(
        NN_seeds,
        XGB_seeds,
        SVM_seeds
    )
    probabilidades, resultado_final = clasificador.clasificar(cluster_features, ensemble_dir, models_dir) 
    #print("Probailidades del clasificador:\n", probabilidades.iloc[:, 1])
    #print("Predicción final:", resultado_final.iloc[:])
    df_confirmed_umbrales["predicted_label"] = resultado_final.iloc[:]

    """df_confirmed_umbrales = df_confirmed_umbrales[
        df_confirmed_umbrales["NAME"] == "BARRY"
    ]"""
    graph_aciertos_hora(df_confirmed_umbrales)
    graph_registros(df_confirmed_umbrales)
    precision, recall, accuracy, far = matriz(df_confirmed_umbrales, lead_time)

    print('\nBest model´s performance:')
    print(f"Precision: {precision:.4f}")
    print(f"Recall (POD):    {recall:.4f} ({(recall*100):.4f})")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"FAR:  {far*100:.4f}%")


if __name__ == "__main__":
    main(color_graph, lead_time)