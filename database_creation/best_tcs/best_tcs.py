#versión que no usa EE

import os
import pandas as pd
import folium
from datetime import timedelta
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from config import start_download_date, stop_download_date


# Cargar los polígonos de prueba
pacific_gdf = gpd.read_file("database_creation/data/shp_files/pacifico_shp/pacifico_shp.shp")
atlantic_gdf = gpd.read_file("database_creation/data/shp_files/atlantico_shp/atlantico_shp.shp")


#definición de métodos
def get_best_tc_maps_and_names(df, sid_txt_path):

    # Union the two areas into one geometry for convenience
    area_geom = pacific_gdf.unary_union.union(atlantic_gdf.unary_union)

    # Prepare output directories
    inside_dir = "database_creation/best_tcs/inside_area"
    outside_dir = "database_creation/best_tcs/outside_area"
    os.makedirs(inside_dir, exist_ok=True)
    os.makedirs(outside_dir, exist_ok=True)


    # Filter: seasons 2023–2025, NAME != 'UNNAMED', Basins no requeridas
    df = df[
        (df["ISO_TIME"] >= pd.Timestamp(start_download_date)) & # Considera los 10 día previos
        (df["ISO_TIME"] <= pd.Timestamp(stop_download_date)) &
        (df["NAME"] != "UNNAMED") &
        (~df["BASIN"].isin(["SI", "SP", "SA", "NI"]))
    ]

    inside_sids = []

    for sid, group in df.groupby("SID"):
        traj = group.sort_values("ISO_TIME")
        coords = list(zip(traj["LAT"], traj["LON"]))
        times = list(traj["ISO_TIME"])
        name = traj["NAME"].iloc[0]
        first_time = times[0]

        # Determine how many points fall inside the unioned area
        inside_count = 0
        prepared_geom = prep(area_geom)
        for lat, lon in coords:
            pt = Point(lon, lat)
            if prepared_geom.contains(pt):
                inside_count += 1
        ratio_inside = inside_count / len(coords)

        # Choose output subdirectory
        if ratio_inside >= 0.4:
            out_subdir = inside_dir
            inside_sids.append(sid)
        else:
            out_subdir = outside_dir

        # Build a folium map centered on the first point
        m = folium.Map(location=[coords[0][0], coords[0][1]], zoom_start=4, tiles="CartoDB positron")

        # Add polygons to the map
        folium.GeoJson(
            pacific_gdf.__geo_interface__,
            name="Pacific Area",
            style_function=lambda feat: {"color": "blue", "fillOpacity": 0.1}
        ).add_to(m)
        folium.GeoJson(
            atlantic_gdf.__geo_interface__,
            name="Atlantic Area",
            style_function=lambda feat: {"color": "green", "fillOpacity": 0.1}
        ).add_to(m)

        # Add the trajectory line
        folium.PolyLine(
            locations=coords,
            color="red",
            weight=2.5,
            opacity=0.8,
            popup=f"{name} ({sid})"
        ).add_to(m)

        # Add a title to the map
        title_html = f"""
            <h3 style="position: absolute; top: 10px; left: 50px; z-index:9999;
                    background-color: white; padding: 5px; border-radius: 3px;">
                {name} — {first_time.strftime('%Y-%m-%d %H:%M')} UTC
            </h3>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        # Save the map
        out_path = os.path.join(out_subdir, f"{sid}.html")
        m.save(out_path)
        print(f"Saved map for {sid} (inside_ratio={ratio_inside:.2%}) to {out_path}")

    # After processing, you have:
    # - Interactive HTML maps in inside_area and outside_area
    # - A list `inside_sids` of all SIDs with ≥70% of points inside
    print("Cyclones with ≥40% of points inside area:")
    for tc in inside_sids:
        print(tc)

    #guardarlos en una lista
    with open(sid_txt_path, 'a', encoding='utf-8') as f:
        if os.path.getsize(sid_txt_path) > 0:
            f.write('\n')  # agrega una línea en blanco si ya hay contenido
        for elemento in inside_sids:
            f.write(f"{elemento}\n")


def clasificar_region_trayectoria(lat_lon_list, pacific_prepared, atlantic_prepared):
    """Devuelve 'pacifico', 'atlantico' o None dependiendo del área predominante"""
    pacific_count = 0
    atlantic_count = 0

    for lat, lon in lat_lon_list:
        pt = Point(lon, lat)
        if pacific_prepared.contains(pt):
            pacific_count += 1
        elif atlantic_prepared.contains(pt):
            atlantic_count += 1

    if pacific_count > atlantic_count:
        return "pacifico"
    else:
        return "atlantico"

def generar_fechas_best_tcs(df, sid_txt_path, output_csv_path):
    #carga los polígnos para cada área
    pacific_geom = pacific_gdf.unary_union
    atlantic_geom = atlantic_gdf.unary_union
    pacific_prepared = prep(pacific_geom)
    atlantic_prepared = prep(atlantic_geom)

    # Leer SIDs del archivo
    with open(sid_txt_path) as f:
        sids = [line.strip() for line in f if line.strip()]

    fechas = []
    for sid in sids:
        df_sid = df[df["SID"] == sid]
        if df_sid.empty:
            continue
    
        coords = list(zip(df_sid["LAT"], df_sid["LON"]))
        region = clasificar_region_trayectoria(coords, pacific_prepared, atlantic_prepared)
        
        inicio = df_sid["ISO_TIME"].min() - timedelta(days=10)
        fin = df_sid["ISO_TIME"].max()
        # Generar fechas con intervalo de 6 horas
        fechas_tc = pd.date_range(start=inicio, end=fin, freq="6h")
        #fechas.extend(fechas_tc)
        fechas.extend([(f, region) for f in fechas_tc])

    # Eliminar duplicados por combinación (fecha, región), ordenar y guardar
    fechas_df = pd.DataFrame(fechas, columns=["fecha", "region"])
    fechas_df = fechas_df.drop_duplicates(subset=["fecha", "region"])
    fechas_df = fechas_df.sort_values("fecha")
    fechas_df.to_csv(output_csv_path, index=False)


# main -------------------------------------
def main():
    sid_txt_path = "database_creation/best_tcs/TC_names_inside.txt"
    output_csv_path = "database_creation/best_tcs/fechas_best_tcs.csv"

    # Read the IBTrACS CSV
    csv_path = "database_creation/data/ibtracs.ALL.list.v04r01.csv"
    df = pd.read_csv(
        csv_path,
        parse_dates=["ISO_TIME"],
        skiprows=[1],  # skip extra header row if present
        usecols=["SID", "NAME", "SEASON", "ISO_TIME", "LAT", "LON", "BASIN"]
    )


    # Genera mapas que contienen los polígonos de las áreas de descarga y las trayectorias de los TCs
    #de los ultimos años. Guarda en carpetas separadas los TC que estuvieron un 40% del tiempo en dichas
    # áreas y los que no. Tamben extrae el nombre de los TC que sí estuvieron y los guarda en un txt.
    get_best_tc_maps_and_names(df, sid_txt_path)

    #usando el txt anterior, genera un csv con las fechas en que ocurrieron los TC - 10 días
    generar_fechas_best_tcs(df, sid_txt_path, output_csv_path)



if __name__ == "__main__":
    main()
