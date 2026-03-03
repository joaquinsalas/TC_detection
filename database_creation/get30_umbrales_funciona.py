# en este código se leerá umbrales_ciclones.csv (ya con la label confirmed) y se dibujarán un par de histogramas de los histogramas que se presentan en cada instante del timepod de vida del ciclón.
# el objetivo es ver cuáles son los umbrales más comunes en cada momento de tiempo
#tenemos que definir las parejas de umbrales (best option)
# o podemos definir los umbrales individualmente y basarme en ello 


# este código tiene la función de recibir umbrales_ciclones.csv que genera umbrales.py,
# quitar las columnas que no son necesarias, agregar la etiqueta 'confirmed' y tenerlo listo para 2 propósitos:
# 1) graficar los histogramas de los umbrales en cada instante
# 2) generar un csv que resume la info de esos umbrales
# 3) Para Entrenar un clasificador

import pandas as pd
#from XGBoost import add_labels
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import ee

plt.style.use("dark_background")
# 2. Definir los valores de umbrales que nos interesan
dist_vals = [400, 300, 200, 100, 50]
min_vals = [5, 10, 15, 20, 30, 40]

def clasificar_region_trayectoria(lat_lon_list, pacific_geom, atlantic_geom):
    """Devuelve 'pacifico', 'atlantico' o None dependiendo del área predominante"""
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

def identifica_zona(df, df_ibtracs):
    # Inicializar Earth Engine
    ee.Initialize()
    #carga los polígnos para cada área
    pacific_fc = ee.FeatureCollection("projects/ee-salas/assets/pacifico")
    atlantic_fc = ee.FeatureCollection("projects/ee-salas/assets/atlantico_shp")
    pacific_geom = pacific_fc.geometry()
    atlantic_geom = atlantic_fc.geometry()

    # agrupar df por SIDs
    resultado = []
    for sid in df["SID"].unique():
        # Obtener todas las coordenadas de ese SID desde df_ibtracs
        df_sid = df_ibtracs[df_ibtracs["SID"] == sid]
        if df_sid.empty:
            continue
        coords = list(zip(df_sid["LAT"], df_sid["LON"]))
        zona = clasificar_region_trayectoria(coords, pacific_geom, atlantic_geom)
        resultado.append({"SID": sid, "zona": zona})

    return pd.DataFrame(resultado)

def grafica_umbrales(heatmap_data):
    # 8. Dibujo del heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        cmap='hot',
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Percentage (%)'},
        #annot=True, fmt=".1f"
        annot=False,
        vmin=0,
        vmax=100,
    )
    plt.tick_params("x", rotation=30)
    plt.xlabel('Hours since the start of the cyclone')
    plt.ylabel('Threshold (distance km, min. trajectories)')
    plt.title('Use of threshold combinations since the start of the cyclone')
    plt.tight_layout()
    plt.savefig('/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/figures_black/heatmap_umbrales_vs_horas_diff.png', dpi=300)


def add_labels(df):
    """
    confirmed: 1 si la fecha en que se hizo la predicción es >= que la fecha oficial del inicio del evento. 0 si es menor
    """
    csv_path = "/home/nathaliealvarez/Personal/databases/ibtracs.last3years.list.v04r01.csv"
    df_ibtracs = pd.read_csv(csv_path, usecols=["ISO_TIME", "SID","USA_SSHS", 'LAT', 'LON'],parse_dates = ['ISO_TIME'])
    # Asegúrate de que las columnas de fechas están en formato datetime
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df_ibtracs['ISO_TIME'] = pd.to_datetime(df_ibtracs['ISO_TIME'], errors='coerce')

    # Obtener fecha de primer registro por SID (inicio oficial del evento)
    df_inicio = (
        df_ibtracs[df_ibtracs["USA_SSHS"] >= 0] # Filtrar solo cuando ya es TS
        .groupby("SID", as_index=False)["ISO_TIME"]
        .min()
        .rename(columns={"ISO_TIME": "inicio_oficial"})
    )
    # Merge de df con fecha de inicio oficial
    df_merged = pd.merge(df, df_inicio, on='SID', how='left')
    # 3) Merge para traer 'USA_SSHS' de IBTrACS en la misma fecha de cada predicción
    df_merged = pd.merge(
        df_merged,
        df_ibtracs[['SID', 'ISO_TIME', 'USA_SSHS']],
        on=['SID', 'ISO_TIME'],
        how='left'
    )

    #agregar la columna de zonas
    zonas_df = identifica_zona(df, df_ibtracs)
    df_merged = pd.merge(df_merged, zonas_df, on='SID', how='left')

    # Agregar columnas de etiquetas
    #si la fecha en la que se hizo la predicción es >= que la fecha del primer registro en IBTrACS
    # y si además el valor reportado de USA_SSHS es >= 0
    # entonces el cluster pronosticado en ese momento se considera un ciclón maduro
    df_merged['confirmed'] = (
        #(df_merged['ISO_TIME'] >= df_merged['inicio_oficial']) &
        (df_merged['USA_SSHS'] >= 0)
    ).astype(int)
    return df_merged

# deja el archivo umbrales_ciclones con las columnas necesarias
def limpiar_archivo_umbrales(path):
    # Leer el archivo CSV
    df = pd.read_csv(path, usecols=["SID","NAME","ISO_TIME","fecha_inicio","distancia_enlace_km","min_trayectorias_por_cluster","n_trayectorias_best_cluster","dispersión_km_best_cluster"],parse_dates = ['ISO_TIME', 'fecha_inicio'])

    df = df[df['n_trayectorias_best_cluster'].notna()]
    df.reset_index(drop=True, inplace=True)

    df = add_labels(df)
    df = df.rename(columns={'ISO_TIME': 'fecha_prediccion', "fecha_inicio": "fecha_estimada_de_inicio","distancia_enlace_km": "umbral_distancia_enlace_km", "min_trayectorias_por_cluster": "umbral_min_trayectorias_por_cluster", "confirmed": "label"})

    # Calcular la cantidad de horas reales entre la fecha de predicción y a fecha en que de verdad inició el ciclón
    df["horas_diff_reales"] = (df["fecha_prediccion"] - df["inicio_oficial"]).dt.total_seconds() / 3600.0

    # Calcular la cantidad de horas estimadas entre la fecha de predicción y la fecha estimada de inicio
    df['horas_diff_estimadas'] = (df['fecha_prediccion'] - df['fecha_estimada_de_inicio']).dt.total_seconds() / 3600.0  # convertir a horas

    #df = df[["SID","NAME","fecha_prediccion","inicio_oficial","horas_diff","umbral_distancia_enlace_km","umbral_min_trayectorias_por_cluster","dispersión_km_best_cluster","n_trayectorias_best_cluster","label"]]

    # Guardar el resultado en un nuevo CSV (o sobrescribir si lo deseas)
    output_path = '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv'
    df.to_csv(output_path, index=False)
    return df


# main -------------------------------------------------------------------------------------------------------
# archivo umbrales_ciclones.csv
path = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/umbrales_ciclones.csv" 
#path = '/home/nathaliealvarez/Personal/umbral_definition/retroalimentaicon_humana/last_updated_umbrales_ciclones.csv' #cambia esto
df = limpiar_archivo_umbrales(path)

output_path = '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv'
#output_path = '/home/nathaliealvarez/Personal/umbral_definition/retroalimentaicon_humana/confirmed_umbrales_ciclones_humano.csv' #cambia esto
df.to_csv(output_path, index=False)

#abrir el archivo limpio
df = pd.read_csv(output_path)


#una vez guardado, se genera la grafica para el analisis de parejas de umbrales a lo largo del tiempo
col_horas_diff = 'horas_diff_estimadas' # diferencia de horas que aparecerán en el eje X
# 3. Filtrar el dataframe a las combinaciones deseadas
df = df[
    df['umbral_distancia_enlace_km'].isin(dist_vals) &
    df['umbral_min_trayectorias_por_cluster'].isin(min_vals)
]

# 4. Crear una etiqueta única para cada pareja de umbrales
df['umbral_combo'] = (
    df['umbral_distancia_enlace_km'].astype(int).astype(str) + ' km, ' +
    df['umbral_min_trayectorias_por_cluster'].astype(int).astype(str) + ' traj.'
)

# 5. Calcular el porcentaje de ocurrencias por combo y horas_diff
#    Asumimos que el total de casos por cada horas_diff es el mismo para todas las combos;
#    si no fuera así, habría que normalizar por el total global de cada horas_diff.
group = df.groupby([col_horas_diff, 'umbral_combo']).size().reset_index(name='count')
total_per_h = group.groupby(col_horas_diff)['count'].transform('sum')
group['pct'] = group['count'] / total_per_h * 100

# 6. Pivot para obtener matriz (Y × X) de porcentajes
heatmap_data = group.pivot(index='umbral_combo', columns=col_horas_diff, values='pct')

# 7. Ordenar filas y columnas si se desea
#    (por ejemplo, ordenar horas_diff de menor a mayor y combos en el orden de dist_vals/min_vals)
heatmap_data = heatmap_data.reindex(
    index=[f"{d} km, {m} traj." for m in min_vals for d in dist_vals],
    columns=sorted(heatmap_data.columns)
)

#descomenta esto cuando ya no estés usando retroalimnetacion humana
# #grafica el heatmap de los umbrales
grafica_umbrales(heatmap_data)


# generar el csv de resumne de mejores umbrales por cada hora ------------------
# ——— AÑADIDO: Guardar CSV con el combo más frecuente por horas_diff ———

# Partimos del DataFrame 'group' que tiene columnas [col_horas_diff, 'umbral_combo', 'count', 'pct']
# 1) Para cada horas_diff, elegir el índice donde 'pct' es máximo
idx_max = group.groupby(col_horas_diff)['pct'].idxmax()

# 2) Extraer sólo las filas ganadoras
best_combo = group.loc[idx_max].reset_index(drop=True)

# 3) Separar 'umbral_combo' en dos columnas numéricas
#    Ejemplo de umbral_combo: "400 km, 3 traj."
best_combo[['umbral_distancia_enlace_km', 'umbral_min_trayectorias_por_cluster']] = (
    best_combo['umbral_combo']
    .str.extract(
        r'(?P<umbral_distancia_enlace_km>\d+(?:\.\d+)?)\s*km,\s*(?P<umbral_min_trayectorias_por_cluster>\d+(?:\.\d+)?)\s*traj\.'
    )
)

# 4) Seleccionar y renombrar las columnas de salida
salida = best_combo[[col_horas_diff,
                     'umbral_min_trayectorias_por_cluster',
                     'umbral_distancia_enlace_km']]

# 5) Guardar en CSV
# descomenta esto cuando ya no estées usando la retroalimentación humana
salida.to_csv(
    '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/umbral_combo_ganador_por_hora.csv',
    index=False
)

print("CSV de combos ganadores guardado en 'umbral_combo_ganador_por_hora.csv'")
