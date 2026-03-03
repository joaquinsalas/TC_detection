#Notas:
# la col 'fecha_inicio' que se obtiene a partir de aquí está desactualizada, se debe extraer desde IBTRACS
# la col. 'label' está desactualizada, se extrae correctamente en otro script

#nota curiosa: dupliqué los registros de 2025146N10264 en el TC_names_insside y por ende también en umbrales_ciclones.csv


from datetime import datetime, timedelta
import os
import re
import pandas as pd
from pathlib import Path
from cluster_analysis import ClusterTCStitchNodes, create_data_folder, dibuja_clusters
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
import ee
from matplotlib.dates import num2date

# Estilo oscuro (no te olvides de cambiar tambie los colores en plot_evolucion())
plt.style.use("dark_background")


base_dir= '/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan'
umbrales_csv_path= f'{base_dir}/umbrales_ciclones_black.csv'
sid_txt_path = f'{base_dir}/TC_names_inside.txt'
tracks_csv_path = "/home/nathaliealvarez/Personal/databases/ibtracs.last3years.list.v04r01.csv"

# Cargar los polígonos
ee.Initialize()
pacific_fc = ee.FeatureCollection("projects/ee-salas/assets/pacifico")
atlantic_fc = ee.FeatureCollection("projects/gencastnathalie/assets/atlantico_extended_v2")

# grid:
valores_distancia = [400, 300, 200, 100, 50]
valores_min_n = [5, 10, 15, 20, 30, 40]


# graficar todas las estadísticas en un gráfico
def plot_evolucion(df_stats, storm_name, fecha_inicio):
    """
    df_stats debe tener columnas:
      - 'date' (datetime)
      - 'n_trajs' (int)
      - 'start_dispersion_km' (float)
    """
    # Copia y reemplaza 0 por NaN
    df = df_stats.copy()
    df["n_trajs"] = df["n_trajs"].replace(0, np.nan)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # ——— Eje izquierdo: Nº de trayectorias (amarillo) ———
    ax1.plot(
        df["date"],
        df["n_trajs"],
        color="yellow", #orange
        marker="o",
        linestyle="-",
        label="Nº trajectories",
    )
    ax1.set_ylabel("Number of trajectories", color="yellow") #black
    ax1.set_ylim(0, 55)
    ax1.tick_params(axis="y", labelcolor="yellow") # black
    ax1.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.3)

    # Configurar ticks X cada 6 horas
    ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 12)))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d_%H:%M"))
    fig.autofmt_xdate(rotation=45)

    # ——— Eje derecho: Dispersión (verde) ———
    ax2 = ax1.twinx()
    ax2.plot(
        df["date"],
        df["start_dispersion_km"],
        color="green",
        marker="o",
        linestyle="-",
        label="Initial dispersion",
    )
    ax2.set_ylabel("Initial dispersion (km)", color="green") #black
    ax2.set_ylim(0, 400)
    ax2.tick_params(axis="y", labelcolor="green") #black (pero el verde clarito)
    ax2.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.3)

    # Línea vertical blanca para la fecha de inicio en IBTrACS --------------------------------------------------
    ax1.axvline(fecha_inicio, color='white', linestyle='--', linewidth=2, label='Cyclogenesis start date') #black

    # Leyenda conjunta
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_xlabel("Date")
    plt.title(f"Number of trajectories and dispersion of predicted clusters for {storm_name}", color="white") #black
    plt.tight_layout()
    plt.savefig(
        f"{base_dir}/figures_black/stats_{storm_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# obtiene los umbrales deseados del cluster
def summarize_best_cluster(best_cluster, results, tool):
    """
    Extrae estadísticas del cluster ganador:
      - n_trajs: número de trayectorias
      - start_coords: [(lat, lon), ...]
      - end_coords: [(lat, lon), ...]
      - start_dispersion_km: distancia media al centroid de inicio
    `tool` es la instancia de ClusterTCStitchNodes (para llamar a _lldistkm).
    """
    trajs = results["trajectories"]
    clusters = results["clusters"]

    # Filtrar sólo las trayectorias del cluster ganador
    best_trajs = [traj for traj, cid in zip(trajs, clusters) if cid == best_cluster]
    n = len(best_trajs)

    # Convertir la primera fecha de la primera trayectoria del cluster
    if best_trajs and "t" in best_trajs[0]:
        traj_start_time = num2date(best_trajs[0]["t"][0]).replace(tzinfo=None)
    else:
        traj_start_time = None

    # Extraer coordenadas de inicio y fin
    start_coords = [(traj["lat"][0], traj["lon"][0]) for traj in best_trajs]
    end_coords = [(traj["lat"][-1], traj["lon"][-1]) for traj in best_trajs]

    start_disp = tool._dispersion_km(start_coords) if start_coords else np.nan
    return {
        "n_trajs": n,
        "start_coords": start_coords,
        "end_coords": end_coords,
        "start_dispersion_km": start_disp,
        "traj_start_time": traj_start_time,
    }


# similar a como lo hacemos con yael
def get_clusters(current, storm_name, cont, stitch_dir, df):
    best_cluster = None
    file_name = (
        f"{current:%Y_%m_%d}_{current.hour:02d}_nodes_ens"  # generado en base a current
    )
    # file_name = '2025_06_10_18_nodes_ens' # generado en base a current
    data_path = os.path.join(base_dir, file_name) 
    create_data_folder(
        data_path, stitch_dir
    )  # crea la carpeta temporal donde se guardan 50 stitchnodes
    file_pattern = "*.txt"
    
    
    if not os.path.isdir(data_path):
        print(f"Error: carpeta '{data_path}' no encontrada.")
    else:
        tool = ClusterTCStitchNodes()

        #results = tool.cluster_tcs(
        #    data_path, file_pattern, distancia_enlace_km, min_trayectorias_por_cluster
        #)
        # -------------------------------------------
        # Grid Search para hallar los mejores umbrales
        # -------------------------------------------
        print(f'grid_search para {storm_name}')
        df_obs = df
        best_params, best_error, best_cluster, best_results = grid_search_umbral(
            tool, data_path, file_pattern, df_obs, valores_distancia, valores_min_n
        ) #if best_error=np.float64(nan), nunca hubo clusteres cuya fecha estimada coincidiera con la fecha observada

        if best_results and best_results["clusters"].size > 0: #si se formó cualquier cluster
            clusters = best_results["clusters"]
            nc = len(np.unique(clusters))
            nt = len(best_results["trajectories"])
            print(f"{nc} clústeres válidos de {nt} trayectorias.")
            # Selecciona el mejor cluster
            best_cluster = best_cluster

    # Borrar carpeta data_path
    shutil.rmtree(data_path)
    #print(f"Carpeta eliminada: {data_path}")

    return best_cluster, best_results, tool, best_params, best_error



# hacer una lista de todos los momentos de tiempo (en saltos de 12hrs) del evento real
# recorrer los stitchnodes desde star:date - 2 dias hasta end_date en paquetes de 50
# leer cda stitch dentro del paquete de 50:
#   buscar que en al menos uno de sus paquetes start tenga al menos una fecha y hora del evento real y que se encuentre a menos de 5 grados de distancia del punto real (provisional)
#   si es así, añade 1 un contador
#   al final debe decir 'en el ensamble del día_hora se encontraron n coincidencias con uno de los puntos de dalila

def data_umbrales(start_date, end_date, stitch_dir, df, storm_name):
    umbrales_list = []
    TIME_STEP = timedelta(hours=6)

    # revisa los paquetes de 50 stitchnodes cada 6 horas y busca coincidencias espaciotemporales
    stats_list = []
    current = start_date
    cont = 0
    while current <= end_date:
        cont += 1
        #run_hr = f"{current.hour:02d}"
        #key = f"{current:%Y%m%d}_{run_hr}"

        # ---------------------------------------------------------------------
        # Los matches espaciales se harán de acuerdo a la separación de clusters de Yael
        # ---------------------------------------------------------------------
        best_cluster, clusters, tool, best_params, best_error = get_clusters(
            current, storm_name, cont, stitch_dir, df
        )#if np.float64(nan)...
        if best_params is not None and clusters is not None: # si hubo culauwier cluster
            d, min_n = best_params
            storm_sid = df["SID"].iloc[0]
           
            # Calcular stats del mejor cluster
            stats = summarize_best_cluster(best_cluster, clusters, tool)
            n_trayectorias_best_cluster = stats["n_trajs"]
            dispersión_km_best_cluster = round(stats["start_dispersion_km"], 2)
            fecha_inicio = stats["traj_start_time"]

            # Extraer el valor de USA_SSHS para el inicio de la trayectoria prevista
            mask = (df["ISO_TIME"] == stats["traj_start_time"])
            if not df[mask].empty:
                usa_ssh_value = df[mask]["USA_SSHS"].iloc[0]
            else:
                usa_ssh_value = np.nan
            #label = 1 if pd.notna(usa_ssh_value) and usa_ssh_value >= 0 else 0

            #si es que no se encontró un cluster cuyas fehcas iniciales emepzaran en el intervalo de tiempo que duró el ciclón, se dejan los campos en blanco
            if best_error is None or np.isnan(best_error):
                d, min_n, n_trayectorias_best_cluster, dispersión_km_best_cluster, usa_ssh_value, fecha_inicio = (np.nan, np.nan, np.nan, np.nan, np.nan, None)

            umbrales_list.append({
                "SID": storm_sid,
                "NAME": storm_name,
                "ISO_TIME": current.strftime("%Y-%m-%d %H:%M:%S"),
                "distancia_enlace_km": d,
                "min_trayectorias_por_cluster": min_n,
                "error_promedio_km": round(best_error, 2),
                "n_trayectorias_best_cluster": n_trayectorias_best_cluster,
                "dispersión_km_best_cluster": dispersión_km_best_cluster,
                "USA_SSHS": usa_ssh_value,
                #"label": label,
                "fecha_inicio": fecha_inicio # Fecha pronosticada de inicio del evento (fecha de inicio de una de las trayectorias del best_cluster)
            })

            # Obtener estadísticas del cluster ganador
            #print(f"Nº de trayectorias: {stats['n_trajs']}")
            #print(f"Dispersión inicios (km): {stats['start_dispersion_km']:.2f}")
            stats_list.append(
                {
                    "date": current,
                    "n_trajs": n_trayectorias_best_cluster,
                    "start_dispersion_km": dispersión_km_best_cluster,
                }
            )
        current += TIME_STEP

    #guardar los umbrales por timestamp deuna sola tormenta
    if umbrales_list:
        df_out = pd.DataFrame(umbrales_list)

        if not os.path.exists(umbrales_csv_path) or os.path.getsize(umbrales_csv_path) == 0:
            pd.DataFrame(columns=[
                "SID", "NAME", "ISO_TIME",
                "distancia_enlace_km", "min_trayectorias_por_cluster", "error_promedio_km", "n_trayectorias_best_cluster", "dispersión_km_best_cluster", "USA_SSHS", "fecha_inicio"
            ]).to_csv(umbrales_csv_path, index=False)

        if os.path.exists(umbrales_csv_path):
            df_existente = pd.read_csv(umbrales_csv_path)

            # Evitar duplicados exactos por SID + ISO_TIME
            df_out = pd.concat([df_existente, df_out], ignore_index=True)
            df_out.drop_duplicates(subset=["SID", "ISO_TIME"], keep="last", inplace=True)

        df_out.to_csv(umbrales_csv_path, index=False)
        print(f"Se guardaron {len(umbrales_list)} registros de umbrales en {umbrales_csv_path}")

    # generar un dataframe de stats para graficar
    df_stats = pd.DataFrame(stats_list)
    if not df_stats.empty: #no grafica si está vacío
        df_stats = df_stats.sort_values("date")
        #start_date = df["ISO_TIME"].min() 
        start_date = df.loc[df["USA_SSHS"] >= 0, "ISO_TIME"].min()
        plot_evolucion(df_stats, storm_name, start_date)
    else:
        print(f"No se encontraron clusters válidos para {storm_name}. No se genera figura.")

    

def clasificar_region_trayectoria(lat_lon_list):
    """Devuelve 'pacifico', 'atlantico' o None dependiendo del área predominante"""
    #carga los polígnos para cada área
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

def grid_search_umbral(tool, data_folder, file_pattern, df_obs, valores_distancia, valores_min_n):
    best_params = None
    best_error = np.inf
    best_cluster_id = None
    best_results = None

    for d in valores_distancia:
        for min_n in valores_min_n:
            try:
                results = tool.cluster_tcs(
                    folder=data_folder,
                    pattern=file_pattern,
                    link_tol_km=d,
                    min_n=min_n
                )

                if np.array(results["clusters"]).size == 0:
                    continue

                best_cluster = tool.select_best_cluster(results, df_obs)
                cluster_trajs = [
                    traj for traj, cid in zip(results["trajectories"], results["clusters"]) if cid == best_cluster
                ]
                if not cluster_trajs:
                    continue

                errors = [
                    tool._mean_error_traj_vs_obs_filtered(traj, df_obs)
                    for traj in cluster_trajs
                ]
                #aqui es donde aparece el error
                mean_error = np.mean([e for e in errors if np.isfinite(e)]) # errors puede ser una lista de inf
                

                # NUEVO: criterio de desempate cuando mean_error == best_error
                is_better = False
                if best_params is None or mean_error < best_error:
                    is_better = True
                elif np.isclose(mean_error, best_error):
                    # desempate: preferir min_n más grande;
                    # si son iguales, preferir d más pequeño
                    _, best_min_n = best_params
                    best_d, _      = best_params
                    if (min_n > best_min_n) or (min_n == best_min_n and d < best_d):
                        is_better = True

                if is_better:
                    best_error      = mean_error
                    best_params     = (d, min_n)
                    best_cluster_id = best_cluster
                    best_results    = results

            except Exception as e:
                print(f"Error con d={d}, min_n={min_n}: {e}")

    return best_params, best_error, best_cluster_id, best_results #if best_error=np.float64(nan), nunca hubo clusteres cuya fecha estimada coincidiera con la fecha observada



# main #############################################
dias_previos = 10
df = pd.read_csv(
    tracks_csv_path,
    usecols=["NAME", "ISO_TIME", "LAT", "LON", "SID", "USA_SSHS"],
    dtype={"LAT": float, "LON": float},
    skiprows=[1],
    parse_dates=["ISO_TIME"],
)

# Leer SIDs del archivo
with open(sid_txt_path) as f:
    sids = [line.strip() for line in f if line.strip()]

for sid in sids:
    df_sid = df[df["SID"] == sid] # sid
    if df_sid.empty:
        continue

    #obtiene la región de donde proviene la trayectoria ['pacifico', 'atlantico']
    coords = list(zip(df_sid["LAT"], df_sid["LON"]))
    zona = clasificar_region_trayectoria(coords)
    #zona = "atlantico"
    stitch_dir = f"/mnt/externo8T/HurricaneData/analisis_maps/stitches/{zona}/nodes_stitch"
    
    # extrae el nombre, la fecha de inicio y fin de la tormenta en cuestión
    storm_name = df_sid["NAME"].iloc[0]
    start_date = df_sid["ISO_TIME"].min() - timedelta(days=dias_previos)
    end_date = df_sid["ISO_TIME"].max()

    #df_sid.set_index("ISO_TIME", inplace=True)  # indexa por tiempo para acelerar los matchs
    data_umbrales(start_date, end_date, stitch_dir, df_sid, storm_name)
