#nota: tal vez el método que crea los clusters tambien  dbeería devolver la fecha de sus puntos iniciales

#tal vez se puede definir un método de searchgrid para busacr entre los umbrales que mejor RMSE proporcionen para cada uno de los ciclones
#con eso se puede definir un umbral
#guardar cada uno de los mejores dispersión y n_trajectories que mejores resultados hayan dado

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from datetime import datetime
import datetime as _dt
import shutil
from collections import defaultdict
import pandas as pd
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
import scipy.io
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class ClusterTCStitchNodes:
    """
    Una clase para leer, agrupar y visualizar trayectorias de ciclones tropicales
    a partir de archivos de TempestExtremes (StitchNodes).
    """

    def __init__(self):
        """Inicializador de la clase."""
        pass

    def save_figure(self, fig, ax, xlab, ylab, zlab, btitle, fn_out, style="normal"):
        """
        Configura el estilo y guarda la figura de la gráfica.
        """
        if style == "normal":
            bg_color, fg_color = "white", "black"
        else:
            bg_color, fg_color = "black", "white"

        fig.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.xaxis.label.set_color(fg_color)
        ax.yaxis.label.set_color(fg_color)
        ax.zaxis.label.set_color(fg_color)
        ax.title.set_color(fg_color)
        ax.tick_params(axis="x", colors=fg_color)
        ax.tick_params(axis="y", colors=fg_color)
        ax.tick_params(axis="z", colors=fg_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg_color)

        ax.set_xlabel(f"${xlab}$", fontsize=20)
        ax.set_ylabel(f"${ylab}$", fontsize=20)
        ax.set_zlabel(f"${zlab}$", fontsize=14)
        ax.set_title(f"${btitle}$", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.zaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))

        output_dir = os.path.dirname(fn_out)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(fn_out, dpi=300, bbox_inches="tight", facecolor=bg_color)
        # print(f"Figura guardada en: {fn_out}")

    def cluster_tcs(
        self, folder=".", pattern="nodes_stitch_*.txt", link_tol_km =300, min_n = 2
    ):
        """
        Agrupa trayectorias y descarta clústeres pequeños.
        Esta versión incluye mejoras estéticas en la gráfica 3D.
        """
        # 1) Leer archivos
        file_paths = glob.glob(os.path.join(folder, pattern))
        all_traj = []
        # print("Iniciando lectura de archivos...")
        for i, fpath in enumerate(file_paths):
            if os.path.getsize(fpath) == 0:
                # print(f"Omitiendo archivo vacío: {os.path.basename(fpath)}")
                continue
            # print(f"Leyendo archivo {os.path.basename(fpath)}...")
            file_id = int(os.path.basename(fpath).split(".")[0][-2:])
            # all_traj contiene fileId, trajId, t, lat, lon
            all_traj.extend(self._read_stitch_file(fpath, file_id))

        # 2-5) Clustering y filtrado (igual que cluster_from_list)
        result = self.cluster_from_list(all_traj, link_tol_km, min_n)
        #result = self.cluster_from_list_by_dispersion(all_traj, link_tol_km, min_n)


        # 6) Gráfica 3D mejorada si hay resultados
        #ahorita no ocupo las graficas, por eso la comentamos
        final_traj = result["trajectories"]
        final_clusters = result["clusters"]
        if len(final_traj) > 0:
            # print("Generando gráfica 3D mejorada...")
            plt.rcParams.update({"font.family": "serif"})
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")
            n_clusters = np.max(final_clusters)
            colors_map = plt.cm.get_cmap("viridis", n_clusters)
            for k, traj in enumerate(final_traj):
                cid = final_clusters[k]
                ax.plot(
                    traj["lon"],
                    traj["lat"],
                    traj["t"],
                    "-o",
                    linewidth=0.8,
                    markersize=2.5,
                    markerfacecolor="none",
                    markeredgewidth=0.8,
                    color=colors_map(cid - 1),
                    label=f"T{k:02d} (C{cid})",
                )
            ax.xaxis.set_pane_color((0, 0, 0, 0))
            ax.yaxis.set_pane_color((0, 0, 0, 0))
            ax.zaxis.set_pane_color((0, 0, 0, 0))
            ax.grid(True, which="both", linestyle=":", linewidth=0.5)
            ax.view_init(elev=7, azim=-12)
            self.save_figure(
                fig,
                ax,
                "longitude",
                "latitude",
                "time",
                "trajectory clustering",
                "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/figures/clusters_py.png",
                style="black",
            )
            plt.close(fig)
        return result

    def cluster_from_list(self, all_traj, link_tol_km=300, min_n=2):
        """
        Igual que cluster_tcs, pero recibe directamente `all_traj`.
        Devuelve {'trajectories': final_traj, 'clusters': final_clusters}.
        """
        n_traj = len(all_traj)
        if n_traj < 2:
            # print("Advertencia: Menos de dos trayectorias para comparar.")
            return {"trajectories": all_traj, "clusters": np.arange(1, n_traj + 1)}

        # print("Calculando matriz de distancias entre trayectorias...")
        dist_matrix = np.zeros((n_traj, n_traj))
        for i in range(n_traj):
            for j in range(i + 1, n_traj):
                if all_traj[i]["fileID"] == all_traj[j]["fileID"]:
                    d = np.inf
                else:
                    d = self._traj_dist(all_traj[i], all_traj[j])
                dist_matrix[i, j] = dist_matrix[j, i] = d

        finite = dist_matrix[np.isfinite(dist_matrix)]
        if finite.size == 0:
            # print("Advertencia: No hay distancias finitas. No se puede clusterizar.")
            return {"trajectories": [], "clusters": []}
        max_d = np.max(finite)
        dist_matrix[~np.isfinite(dist_matrix)] = max_d * 2

        # print("Realizando clustering jerárquico...")
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method="single")
        raw_clusters = fcluster(Z, t=link_tol_km, criterion="distance")

        unique_lbls, counts = np.unique(raw_clusters, return_counts=True)
        valid = unique_lbls[counts >= min_n]
        label_map = {old: new for new, old in enumerate(valid, start=1)}

        final_traj = []
        final_clusters = []
        for idx, traj in enumerate(all_traj):
            lbl = raw_clusters[idx]
            if lbl in label_map:
                final_traj.append(traj)
                final_clusters.append(label_map[lbl])

        if not final_traj:
            # print("No se conservaron clústeres con el tamaño mínimo.")
            return {"trajectories": [], "clusters": []}

        final_clusters = np.array(final_clusters)
        # print(f"Trayectorias conservadas: {len(final_traj)}")
        # print(f"Clústeres conservados: {len(label_map)} (min_n={min_n})")
        return {"trajectories": final_traj, "clusters": final_clusters}

    def _read_stitch_file(self, fname, file_id):
        """Lee un archivo de nodos y devuelve una lista de trayectorias."""
        with open(fname, "r") as f:
            lines = f.readlines()
        trajectories, t, lat, lon = [], [], [], []
        traj_id_counter = 0

        def save_prev():
            nonlocal traj_id_counter
            if t:
                traj_id_counter += 1
                trajectories.append(
                    {
                        "fileID": file_id,
                        "trajID": traj_id_counter,
                        "t": np.array(t),
                        "lat": np.array(lat),
                        "lon": np.array(lon),
                    }
                )
                return True
            return False

        for line in lines:
            cl = line.strip()
            if not cl:
                continue
            if cl.lower().startswith("start"):
                if save_prev():
                    t, lat, lon = [], [], []
                continue
            try:
                parts = [float(p) for p in cl.split()]
                if len(parts) < 11:
                    continue
                lon_i, lat_i = parts[2], parts[3]
                yy, mm, dd, hh = map(int, parts[7:11])
                dt = datetime(yy, mm, dd, hh)
                t_i = mdates.date2num(dt)
                lon.append(lon_i)
                lat.append(lat_i)
                t.append(t_i)
            except:
                continue
        save_prev()
        return trajectories

    def _traj_dist(self, a, b):
        """Calcula la distancia entre los puntos iniciales de dos trayectorias si ocurren al mismo tiempo."""
        # fechas de incio de las trayectorias a comparar)
        t0_a = a['t'][0]
        t0_b = b['t'][0]
        # Verificar si los tiempos iniciales coinciden (tolerancia de 1 hpra que creo que se puede quitar pero no afecta)
        #if not np.isclose(t0_a, t0_b, atol=1/24):
            #return np.inf
        if abs(t0_a-t0_b) <= 0:
            # calcular la distancia entre los puntos iniciales
            p1 = np.array([[a['lat'][0], a['lon'][0]]])
            p2 = np.array([[b['lat'][0], b['lon'][0]]])
            return self._lldistkm(p1, p2)[0]
        else: 
            return np.inf

    def _lldistkm(self, p1, p2):
        """Distancia Haversine vectorizada."""
        R = 6371 # radio de la Tierra km
        lat1, lon1 = np.radians(p1[:, 0]), np.radians(p1[:, 1])
        lat2, lon2 = np.radians(p2[:, 0]), np.radians(p2[:, 1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    # métodos agregados por Nath
    # para calcular el error promedio entre una trayectoria del ensamble y la observada
    def _mean_error_traj_vs_obs_filtered(self, traj, df_obs, only_first_points=True):
        """
        Calcula el error promedio (km) entre una trayectoria de ensamble y la observada (df_obs).
        - Si only_first_points=False (por defecto): usa todos los timestamps comunes.
        - Si only_first_points=True: usa solo el PRIMER punto de la trayectoria del ensamble (el más reciente)
        y su correspondiente punto observado en el mismo instante.
        Devuelve np.inf si no hay matching temporal.
        """
        # 1) Convertir los floats de tiempo a datetimes
        dt_nums = traj["t"]  # array de matplotlib datenums
        dt_list = [mdates.num2date(t).replace(tzinfo=None) for t in dt_nums]

        # 2) Filtrar df_obs a esos timestamps
        if only_first_points: # si es que solo usaremos el punto inicial
            df_sub = df_obs[df_obs["ISO_TIME"] == dt_list[0]]
        else:
            df_sub = df_obs[df_obs["ISO_TIME"].isin(dt_list)]
        if df_sub.empty:
            return np.inf  # no hay puntos comunes → penalizamos

        # 3) Para cada punto común, emparejar predicción vs observación
        preds, obs = [], []
        for _, row in df_sub.iterrows():
            t = row["ISO_TIME"]
            idx = dt_list.index(t)
            preds.append((traj["lat"][idx], traj["lon"][idx]))
            obs.append((row["LAT"], row["LON"]))

        p1 = np.array(preds)  # [[lat1, lon1], [lat2, lon2], …]
        p2 = np.array(obs)
        dists = self._lldistkm(p1, p2)  # array de distancias en km
        return np.mean(dists)

    # devuelve el cluster cuyas trayectoias se parecen más a la trayectiria observada
    def select_best_cluster(self, results, df_obs):
        """
        Dado `results` (output de cluster_tcs o cluster_from_list) y
        `df_obs` con columnas [NAME, ISO_TIME (datetime), LAT, LON],
        devuelve el cluster cuyo error promedio (sobre todos sus trayectorias
        vs df_obs filtrado) es el menor.
        """
        trajs = results["trajectories"]
        clusters = results["clusters"]

        # agrupar trayectorias por cluster
        agrup = defaultdict(list)
        for traj, cid in zip(trajs, clusters):
            agrup[cid].append(traj)

        errores_por_cluster = {}
        for cid, lista in agrup.items():
            errs = []
            for traj in lista:
                e = self._mean_error_traj_vs_obs_filtered(traj, df_obs, True)
                if np.isfinite(e):
                    errs.append(e)
            errores_por_cluster[cid] = np.mean(errs) if errs else np.inf

        best = min(errores_por_cluster, key=errores_por_cluster.get) 
        # print("Error medio por cluster:", errores_por_cluster)
        # print("→ Cluster ganador:", best)
        return best # best puede ser un diccionario con toods sus valores inf, asíq ue agarrará el primer cluster que guuarde errores_por_cluster

    def _dispersion_km(self, latlon_pts):
        """
        latlon_pts: array-like de shape (n, 2) con [lat, lon] en grados.
        Devuelve la distancia media (km) de cada punto al centroide.
        """
        pts = np.asarray(latlon_pts, dtype=float)
        if pts.size == 0:
            return np.inf
        # centroide en lat,lon (en grados)
        centroid = pts.mean(axis=0)
        centroids = np.tile(centroid, (pts.shape[0], 1))
        return float(np.mean(self._lldistkm(pts, centroids)))

    #por ahora queda en cuarentena
    def cluster_from_list_by_dispersion(self, all_traj, max_disp_km, min_n):
        """
        Forma clusters fusionando grupos si, al unirlos, la dispersión
        (media al centroide de los puntos iniciales) ≤ max_disp_km.
        Solo compara trayectorias que inician (≈) al mismo tiempo.
        """
        if len(all_traj) == 0:
            return {"trajectories": [], "clusters": np.array([])}

        # 1) Agrupar por tiempo inicial (t0) con tolerancia de 1 hora
        groups = defaultdict(list)
        for traj in all_traj:
            t0 = traj["t"][0]
            # cuantiliza a resolución de hora para agrupar
            t0_bucket = np.round(t0 * 24) / 24.0
            groups[t0_bucket].append(traj)

        final_traj = []
        final_clusters = []
        next_label = 1

        # 2) Para cada grupo temporal, aglomerar por dispersión
        for _, trajs in groups.items():
            # Cada trayectoria empieza como su propio cluster
            clusters = [[tr] for tr in trajs]

            # Greedy: mientras exista un par que al fusionarse cumpla dispersión ≤ umbral
            merged = True
            while merged and len(clusters) > 1:
                merged = False
                best_pair = None
                best_disp = None

                # buscar el par que produzca menor dispersión y que ≤ max_disp_km
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        # opcional: evita mezclar trayectorias del mismo archivo, si deseas replicar tu old rule
                        if any(a["fileID"] == b["fileID"] for a in clusters[i] for b in clusters[j]):
                            continue

                        pts = []
                        for tr in (clusters[i] + clusters[j]):
                            lat0 = float(tr["lat"][0])
                            lon0 = float(np.mod(tr["lon"][0], 360))  # 0–360
                            pts.append([lat0, lon0])

                        disp = self._dispersion_km(np.array(pts))
                        if disp <= max_disp_km and (best_disp is None or disp < best_disp):
                            best_disp = disp
                            best_pair = (i, j)

                if best_pair is not None:
                    i, j = best_pair
                    clusters[i].extend(clusters[j])
                    del clusters[j]
                    merged = True

            # 3) Filtrar por tamaño mínimo y volcar a la salida
            for cl in clusters:
                if len(cl) >= min_n:
                    for tr in cl:
                        final_traj.append(tr)
                        final_clusters.append(next_label)
                    next_label += 1

        return {"trajectories": final_traj, "clusters": np.array(final_clusters)}

# definición de métodos
# crea una carpeta temporal donde guarda una copia de los stitchnodes de interés basandose en la fecha
def create_data_folder(folder_path, base_txt_folder):
    #base_txt_folder = "/mnt/externo8T/HurricaneData/processed_trajectories"  # donde se guardan todos los stitch oficiales

    # Preparar nombres
    partes = folder_path.split("/")[-1].split("_")
    timestamp_txt = "".join(partes[:3]) + "_" + partes[3]

    # Crear carpeta
    os.makedirs(folder_path, exist_ok=True)
    ##print(f"Carpeta creada: {folder_path}")

    # Buscar archivos con el patrón nodes_stitch_20240910_06_05.txt
    patron = os.path.join(base_txt_folder, f"*{timestamp_txt}_??.txt")
    #patron = os.path.join(base_txt_folder, f"nodes_stitch_{timestamp_txt}_pacifico_??.txt")
    archivos = glob.glob(patron)
    ##print(f"Archivos encontrados: {len(archivos)}")

    # Copiar archivos
    for archivo in archivos:
        shutil.copy(archivo, folder_path)
        ##print(f"Copiado: {archivo} → {folder_path}")

# imprime las trayectorias de cada uno de los clusters
def imprimir_trayectorias_por_cluster(results):
    from collections import defaultdict

    trajs = results["trajectories"]
    clusters = results["clusters"]
    agrupadas = defaultdict(list)

    for traj, cluster_id in zip(trajs, clusters):
        # Combina fileID y trajID como identificador único
        agrupadas[cluster_id].append((traj["fileID"], traj["trajID"]))

    for cid in sorted(agrupadas):
        print(f"\nCluster {cid} ({len(agrupadas[cid])} trayectorias):")
        for fid, tid in agrupadas[cid]:
            print(f"  - Trayectoria ID {tid} del escenario {fid}")

def dibuja_clusters(best_cluster, results, image_path, storm_name, fecha):
    plt.rcParams.update(
        {
            "text.usetex": True,  # Usa LaTeX para todo el texto
            "font.family": "serif",  # Fuente serif (tipo LaTeX)
            "font.serif": ["Computer Modern"],  # Fuente clásica de LaTeX
            "axes.labelsize": 14,  # Tamaño de etiquetas
            "font.size": 14,  # Tamaño general
        }
    )

    # ---------------------------
    # Cargar y procesar la máscara
    # ---------------------------
    mask_path = "/home/nathaliealvarez/Personal/databases/cygnss_mask.mat"
    mask_data = scipy.io.loadmat(mask_path)
    mask = mask_data.get("cygnss_mask")

    # Invertir los valores de la máscara (0s por 1s y 1s por 0s)
    mask = 1 - mask
    # Voltear la máscara verticalmente
    mask = np.flipud(mask)

    # Definir los límites de la región de interés
    lat_min, lat_max = 0, 35
    lon_min, lon_max = 230, 280

    # Recortar la máscara a la región de interés
    mask_lat_indices = np.linspace(-40, 40, mask.shape[0])
    mask_lon_indices = np.linspace(0, 360, mask.shape[1])
    lat_mask_mask = (mask_lat_indices >= lat_min) & (mask_lat_indices <= lat_max)
    lon_mask_mask = (mask_lon_indices >= lon_min) & (mask_lon_indices <= lon_max)
    mask_cropped = mask[np.ix_(lat_mask_mask, lon_mask_mask)]

    # Crear figura y ejes
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.imshow(
        mask_cropped,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="lower",
        cmap="gray",
        alpha=0.5,
    )

    # --- Dibujar trayectorias ---
    trajs = results["trajectories"]
    clusters = results["clusters"]

    unique_clusters = sorted(set(clusters))
    # Generar colores únicos (excepto el rojo reservado para el mejor)
    cmap = cm.get_cmap("tab20", len(unique_clusters))
    cluster_colors = {
        cid: mcolors.to_hex(cmap(i)) for i, cid in enumerate(unique_clusters)
    }

    # dentro de dibuja_clusters, donde recorres las trayectorias:
    for traj, cid in zip(trajs, clusters):
        # convierto a 0–360
        lons = np.mod(traj["lon"], 360)
        lats = traj["lat"]
        color = cluster_colors.get(cid, "gray")
        ax.plot(lons, lats, color=color, linewidth=0.7, alpha=0.6)

    # idem para el cluster ganador
    if best_cluster is not None:
        for traj, cid in zip(trajs, clusters):
            if cid == best_cluster:
                lons = np.mod(traj["lon"], 360)
                lats = traj["lat"]
                ax.plot(lons, lats, color="red", linewidth=1.2, alpha=0.9)

    # Crear leyenda con los IDs de cluster y su color
    handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=f"Cluster {cid}")
        for cid, color in cluster_colors.items() if cid != best_cluster
    ]
    if best_cluster is not None:
        handles.append(
            plt.Line2D([0], [0], color="red", lw=2, label=f"Cluster {best_cluster} (mejor)")
    )
    ax.legend(handles=handles, loc="best", fontsize=8)

    # Etiquetas y límites
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(f"Clusters de trayectorias de {storm_name} {fecha}")

    # Guardar figura
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def dibuja_poligonos_clusters(self, results, image_path, storm_name, fecha):
    """
    Dibuja, para cada cluster detectado en `results`:
      - El polígono (envolvente convexa) formado por los puntos iniciales.
      - Las trayectorias del cluster en el mismo color del polígono.
      - Una leyenda con la distancia media al centroide del polígono (dispersión, km).
    """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "axes.labelsize": 14,
            "font.size": 14,
        }
    )

    # ---------------------------
    # Cargar y procesar la máscara
    # ---------------------------
    mask_path = "/home/nathaliealvarez/Personal/databases/cygnss_mask.mat"
    mask_data = scipy.io.loadmat(mask_path)
    mask = mask_data.get("cygnss_mask")

    mask = 1 - mask          # invertir
    mask = np.flipud(mask)   # voltear vertical

    # Región de interés
    lat_min, lat_max = 0, 35
    lon_min, lon_max = 230, 280

    # Recorte de máscara
    mask_lat_indices = np.linspace(-40, 40, mask.shape[0])
    mask_lon_indices = np.linspace(0, 360, mask.shape[1])
    lat_mask_mask = (mask_lat_indices >= lat_min) & (mask_lat_indices <= lat_max)
    lon_mask_mask = (mask_lon_indices >= lon_min) & (mask_lon_indices <= lon_max)
    mask_cropped = mask[np.ix_(lat_mask_mask, lon_mask_mask)]

    # Figura
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.imshow(
        mask_cropped,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="lower",
        cmap="gray",
        alpha=0.5,
    )

    # ---------------------------
    # Preparar datos de clusters
    # ---------------------------
    trajs = results["trajectories"]
    clusters = results["clusters"]
    if len(trajs) == 0 or len(clusters) == 0:
        plt.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    unique_cids = sorted(np.unique(clusters))
    cmap = plt.cm.get_cmap("tab20", len(unique_cids))

    legend_handles = []

    #no necesitas volver a declararlo, verdad?
    """# Pequeño helper: dispersión media al centroide (km)
    def _dispersion_km(latlon_pts):
        pts = np.asarray(latlon_pts, dtype=float)
        if pts.size == 0:
            return np.nan
        centroid = pts.mean(axis=0)                           # [lat, lon]
        centroids = np.tile(centroid, (pts.shape[0], 1))
        return float(np.mean(self._lldistkm(pts, centroids))) # usa tu Haversine"""

    for idx, cid in enumerate(unique_cids):
        color = cmap(idx)

        # ---- Puntos iniciales del cluster (lat, lon en grados) ----
        init_pts = []
        cluster_trajs_idx = []
        for k, (traj, lab) in enumerate(zip(trajs, clusters)):
            if lab == cid:
                lat0 = float(traj["lat"][0])
                lon0 = float(np.mod(traj["lon"][0], 360.0))   # a 0–360
                init_pts.append([lat0, lon0])
                cluster_trajs_idx.append(k)

        init_pts = np.array(init_pts)
        if init_pts.shape[0] == 0:
            continue

        # ---- Dispersión (distancia media al centroide) ----
        tool = ClusterTCStitchNodes()
        disp_km = tool._dispersion_km(init_pts)
        #disp_km = _dispersion_km(init_pts)

        # ---- Dibujar trayectorias del cluster en 'color' ----
        for k in cluster_trajs_idx:
            traj = trajs[k]
            lons = np.mod(traj["lon"], 360.0)
            lats = traj["lat"]
            ax.plot(lons, lats, color=color, linewidth=1.0, alpha=0.95)

        # ---- Dibujar polígono de puntos iniciales ----
        # Si hay 3+ puntos, usamos la envolvente convexa; si no, solo scatter
        lons0 = init_pts[:, 1]
        lats0 = init_pts[:, 0]

        if init_pts.shape[0] >= 3:
            try:
                # ConvexHull espera (x, y) = (lon, lat)
                hull = ConvexHull(np.c_[lons0, lats0])
                poly_lon = lons0[hull.vertices]
                poly_lat = lats0[hull.vertices]
                ax.fill(poly_lon, poly_lat, facecolor=color, alpha=0.25, edgecolor=color, linewidth=2)
            except Exception:
                # fallback a dispersión de puntos
                ax.scatter(lons0, lats0, s=25, color=color, alpha=0.9, zorder=3)
        else:
            ax.scatter(lons0, lats0, s=30, color=color, alpha=0.9, zorder=3)

        # ---- Centroid marker ----
        centroid_lat = np.mean(lats0)
        centroid_lon = np.mean(lons0)
        ax.plot(centroid_lon, centroid_lat, marker="x", markersize=8, mew=2, color=color, zorder=4)

        # ---- Leyenda (un handle por cluster) ----
        legend_handles.append(
            Line2D([0], [0], color=color, lw=4, label=f"C{cid}: {disp_km:.1f} km")
        )

    # Decoración
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(f"Polígonos de puntos iniciales por clúster — {storm_name} {fecha}")

    # Leyenda tipo lista en esquina inferior izquierda
    leg = ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.85,
        fontsize=10,
        title="Dispersión media (km)",
        title_fontsize=11,
    )
    leg.get_frame().set_edgecolor("black")

    # Guardar figura
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



# main ------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Modifica las siguientes variales dependiendo de la tormenta que analices:
    storm_name = 'FLOSSIE'#"ALBERTO"
    zona = 'pacifico'#'atlantico'
    instante_str = '2025_10_15_12'#'2024_06_18_00'
    #--------------------------------------------------------------------------------------------------------------------
    data_folder = f"/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/{instante_str}_nodes_ens"
    #base_txt_folder = f'/mnt/externo8T/HurricaneData/analisis_maps/stitches/{zona}/nodes_stitch'
    base_txt_folder = f'/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/analisis_maps'


    create_data_folder(data_folder, base_txt_folder)
    file_pattern = "*.txt"
    # los valores 600km y 3 trayectorias son un aproximado exagerado del numerp d  
    distancia_enlace_km = 200 
    min_trayectorias_por_cluster = 3
    if not os.path.isdir(data_folder):
        print(f"Error: carpeta '{data_folder}' no encontrada.")
    else:
        tool = ClusterTCStitchNodes()
        results = tool.cluster_tcs(
            data_folder, file_pattern, distancia_enlace_km, min_trayectorias_por_cluster
        )
        if results and results["clusters"].size > 0:
            nc = len(np.unique(results["clusters"]))
            nt = len(results["trajectories"])
            # print(f"{nc} clústeres válidos de {nt} trayectorias.")

            # imprimir las trayectorias que corresponden a cada cluster
            imprimir_trayectorias_por_cluster(results)

            #esto lo descomentamos cuando queramos analizar trayectorias de las que haya registros (por eso best_clusster=None)
            best_cluster = None
            storm_name='desconocido'
            """# Detectar el cluster cuyas trayectorias sean más parecidas a la observada --------------------------------
            # abrir el dataframe con los datos de una tormenta
            df = pd.read_csv(
                "/home/nathaliealvarez/Personal/databases/ibtracs.last3years.list.v04r01.csv",
                usecols=["NAME", "ISO_TIME", "LAT", "LON"],
                dtype={"LAT": float, "LON": float},
                skiprows=[1],
                parse_dates=["ISO_TIME"],
            )
            df = df[(df["NAME"] == storm_name)]
            # Selecciona el mejor cluster
            best_cluster = tool.select_best_cluster(results, df)"""

            #ahorita no dibujamos porque no queremos
            image_path = os.path.join(
                "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/figures",
                "mapa_clusters.png",
            )
            #fecha = data_folder.split("/")[-1].split("_")[:4]
            fecha='hoy'
            dibuja_clusters(best_cluster, results, image_path, storm_name, fecha)

        else:
            print("No se formaron clústeres válidos.")

    # Borrar carpeta data_folder
    shutil.rmtree(data_folder)

