#Este código recibe las caracterpisticas de un cluster, procesa los datos y se los pasa al clasificador ensamble para obtener un resultado final

import pandas as pd
from autogluon.tabular import TabularPredictor
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import re
import cv2
import ee

from datetime import datetime, timedelta

tracks_csv_path = "/home/nathaliealvarez/Personal/databases/ibtracs.last3years.list.v04r01.csv"
confirmed_umbrales_path = '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv'

# --- Red neuronal en PyTorch ---
class FlexibleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, activation_name):
        super().__init__()
        layers = []
        prev_dim = input_dim
        # Elige activación según nombre
        if activation_name == "relu":
            activation = nn.ReLU()
        elif activation_name == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name == "tanh":
            activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation_name}")

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # output
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ClasificadorHurakan:
    NN_seeds=[17,3,11] #modelos 0, 1, 2
    XGB_seeds=[0,4,19] #modelos 3, 4, 5
    SVM_seeds=[4,28,26] #modelos 6, 7, 8
    models_seeds = [NN_seeds, XGB_seeds, SVM_seeds]
    names = ['NN', 'XGB', 'SVM']
    def __init__(self):
        pass

    def nn_bundle_predict_proba(self, nn_bundle, X_test):
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

    def svm_bundle_predict_proba(self, svm_bundle, X_test):
        scaler = svm_bundle["scaler"]
        model = svm_bundle["model"]

        # Asegura numpy 2D
        X_np = X_test.values if hasattr(X_test, "values") else X_test
        Xs = scaler.transform(X_np)
        probas = model.predict_proba(Xs)[:, 1]
        return probas

    # --- Helper para cargar modelos y obtener probas ---
    def predict_with_model(self, model_info, X):
        # Si es un bundle NN (dict), reconstruimos; si no, asumimos sklearn (XGB Pipeline)
        if isinstance(model_info, dict) and "model_state" in model_info and "scaler" in model_info:
            probas = self.nn_bundle_predict_proba(model_info, X)
        # Si es un bundle SVM
        elif isinstance(model_info, dict) and "model" in model_info and "scaler" in model_info:
            probas = self.svm_bundle_predict_proba(model_info, X)
        # sirve para XGB
        elif "sklearn.pipeline" in str(type(model_info)):
            # predict_proba(X) ya incluye el escalado
            probas = model_info.predict_proba(X)[:, 1]
        else:
            print('tipo de modelo no reconocido')
            probas = None
        return probas

    def preprocess_input(self, df, models_dir):
        files = []
        #usar los clasificadores NN, XGB y SVM para generar el dataset stackeado
        for model_seed_list, name in zip(self.models_seeds, self.names):
            for seed in model_seed_list:
                #cargar el modelo correspondiente
                file = os.path.join(models_dir, f"{name}_classifier_seed_{seed}.pkl")
                files.append(file)

        # 3. Calcula features stacking: proba de cada modelo base
        preds = pd.DataFrame()
        for idx, f in enumerate(files):
            with open(f, "rb") as pf:
                model_info = pickle.load(pf)
            col_name = f"model_{idx}"
            preds[col_name] = self.predict_with_model(model_info, df)

        return preds


    def clasificar(self, df, ensemble_dir, models_dir):
        #primero obtiene las probabilidades de cada clasificador base
        preds = self.preprocess_input(df, models_dir)

        #cargar el modelo desde model_dir
        predictor = TabularPredictor.load(ensemble_dir)
        probabilidades = predictor.predict_proba(preds)
        resultado_final = predictor.predict(preds)
        return probabilidades, resultado_final
    
def generar_video_cronologico(input_folder, video_name, fps=3):
    """
    Genera un video a partir de imágenes PNG en input_folder.
    Las imágenes deben tener nombres como 'mapa_clusters_YYYY_MM_DD_HH.png'.
    Se ordenan cronológicamente según el nombre. Si falta alguna imagen, se salta.
    """
    # Regex para extraer la fecha y hora del nombre
    pattern = re.compile(r"mapa_clusters_(\d{4}_\d{2}_\d{2}_\d{2})\.png")
    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    
    # Extrae fecha-hora para ordenar
    images_dt = []
    for img in images:
        match = pattern.match(img)
        if match:
            images_dt.append((img, match.group(1)))
    if not images_dt:
        print("No hay imágenes válidas en la carpeta.")
        return

    # Ordena por fecha-hora (alfabético funciona para este formato)
    images_dt.sort(key=lambda x: x[1])
    sorted_imgs = [img for img, _ in images_dt]

    # Verifica que al menos una imagen pueda ser leída
    first_frame = None
    for img in sorted_imgs:
        img_path = os.path.join(input_folder, img)
        frame = cv2.imread(img_path)
        if frame is not None:
            first_frame = frame
            break
    if first_frame is None:
        print("No se pudo leer ninguna imagen.")
        return

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"{input_folder}/{video_name}", fourcc, fps, (width, height))

    for img in sorted_imgs:
        img_path = os.path.join(input_folder, img)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Saltando {img_path}, no se pudo leer.")
            continue
        video.write(frame)

    video.release()
    print(f"Video guardado como {video_name}")

def clasificar_region_trayectoria(lat_lon_list):
    """Devuelve 'pacifico', 'atlantico' o None dependiendo del área predominante"""
    #carga los polígnos para cada área
    ee.Initialize()
    pacific_fc = ee.FeatureCollection("projects/ee-salas/assets/pacifico")
    atlantic_fc = ee.FeatureCollection("projects/ee-salas/assets/atlantico_shp")
    pacific_geom = pacific_fc.geometry()
    atlantic_geom = atlantic_fc.geometry()

    total = len(lat_lon_list)
    pacific_count = 0
    atlantic_count = 0

    for lat, lon in lat_lon_list:
        point = ee.Geometry.Point([lon, lat])
        if pacific_geom.contains(point).getInfo():
            pacific_count += 1
        elif atlantic_geom.contains(point).getInfo():
            atlantic_count += 1

    if pacific_count > atlantic_count:
        return "pacifico"
    else:
        return "atlantico"

#todavía tengo que aplicar la clusterización (similar a comola hice en el video de los 4 paneles)
#de este método solo voy a sacar las parejas de umbrales utilizados en cada momento para clusterizar después
def extract_features(storm_id):
    dias_previos = 10
    df_ibtracs = pd.read_csv(
        tracks_csv_path,
        usecols=["NAME", "ISO_TIME", "SID", "LAT", "LON"],
        skiprows=[1],
        parse_dates=["ISO_TIME"],
    )
    df_confirmed_umbrales = pd.read_csv(
        confirmed_umbrales_path,
        usecols=["fecha_prediccion", "SID", "umbral_min_trayectorias_por_cluster", "umbral_distancia_enlace_km"],
        parse_dates=["fecha_prediccion"],
    )

    df_sid = df_ibtracs[df_ibtracs["SID"] == storm_id] # sid
    #obtiene la región de donde proviene la trayectoria ['pacifico', 'atlantico']
    coords = list(zip(df_sid["LAT"], df_sid["LON"]))
    #zona = clasificar_region_trayectoria(coords)
    zona = 'atlantico'

    # extrae el nombre, la fecha de inicio y fin de la tormenta en cuestión
    storm_name = df_sid["NAME"].iloc[0]
    fecha_inicio = df_sid["ISO_TIME"].min() - timedelta(days=dias_previos)
    fecha_fin = df_sid["ISO_TIME"].max()

    #hacer el video timelapse dle clasificador aplicado a un TC
    from cluster_analysis_clasif_video import main_clasif_video
    main_clasif_video(zona, fecha_inicio, fecha_fin, df_confirmed_umbrales, storm_name) # ya se hizo, descomentar si es la primera vez

    generar_video_cronologico("/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/figures_black/mapas_clasif", f"clasif_timelapse_{storm_name}.mp4", fps=2)





def main(storm_id):
    ensemble_dir = '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos/Gluon/ensemble_seed_2'
    models_dir = '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos_ensamble'

    #Extraer 
    if storm_id: 
        # es una aplicación con uan tormenta real y un video que documenta su progreso y el clasificador sobre ella
        cluster_features = extract_features(storm_id)
    else:
        # Es una aplicación de ejemplo con un solo cluster
        cluster_features = {"n_trayectorias_best_cluster": [10], "dispersión_km_best_cluster": [50], "horas_diff_estimadas": [-12]}
    
        df = pd.DataFrame(cluster_features)

        #crea el objeto
        clasificador = ClasificadorHurakan()
        probabilidades, resultado_final = clasificador.clasificar(df, ensemble_dir, models_dir) 
        print("Probailidades del clasificador:\n", probabilidades)
        print("Predicción final:", resultado_final)



if __name__ == "__main__":
    storm_id = '2024181N09320' #BERYL
    main(storm_id)