from pathlib import Path
import pandas as pd

# =========================
# Paths
# =========================

CSV_PATH = Path(
    "/home/nathaliealvarez/Personal/Repos/TC_detection/database_creation/best_tcs/fechas_best_tcs.csv"
)

BASE_DIRS = {
    "pacifico": Path("/mnt/hurakan-frontend/hurakan/data/nodes_stitch/pacifico"),
    "atlantico": Path("/mnt/hurakan-frontend/hurakan/data/nodes_stitch/atlantico"),
}

OUTPUT_MISSING = CSV_PATH.parent / "missing_nodes_stitch_files.csv"
OUTPUT_FOUND = CSV_PATH.parent / "found_nodes_stitch_files.csv"


# =========================
# Load CSV
# =========================

df = pd.read_csv(CSV_PATH)

# Limpiar nombres de columnas por si tienen espacios
df.columns = df.columns.str.strip().str.lower()

required_columns = {"fecha", "region"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"El CSV no contiene las columnas requeridas: {missing_columns}")

# Normalizar datos
df["fecha"] = pd.to_datetime(df["fecha"])
df["region"] = df["region"].astype(str).str.strip().str.lower()


# =========================
# Check files
# =========================

found_records = []
missing_records = []

for idx, row in df.iterrows():
    fecha = row["fecha"]
    region = row["region"]

    if region not in BASE_DIRS:
        missing_records.append({
            "row": idx,
            "fecha": fecha,
            "region": region,
            "expected_pattern": None,
            "reason": "Región no reconocida"
        })
        continue

    folder = BASE_DIRS[region]

    if not folder.exists():
        missing_records.append({
            "row": idx,
            "fecha": fecha,
            "region": region,
            "expected_pattern": None,
            "reason": f"La carpeta no existe: {folder}"
        })
        continue

    # Formato usado en el nombre del archivo
    date_str = fecha.strftime("%Y%m%d")
    hour_str = fecha.strftime("%H")

    # Ejemplo:
    # nodes_stitch_20260224_12_pacifico_43.txt
    pattern = f"nodes_stitch_{date_str}_{hour_str}_{region}_[0-9][0-9].txt"

    matches = sorted(folder.glob(pattern))

    if matches:
        found_records.append({
            "row": idx,
            "fecha": fecha,
            "region": region,
            "expected_pattern": pattern,
            "n_files_found": len(matches),
            "files_found": ";".join(str(m) for m in matches)
        })
    else:
        missing_records.append({
            "row": idx,
            "fecha": fecha,
            "region": region,
            "expected_pattern": pattern,
            "reason": "No se encontró ningún archivo correspondiente"
        })


# =========================
# Save reports
# =========================

found_df = pd.DataFrame(found_records)
missing_df = pd.DataFrame(missing_records)

found_df.to_csv(OUTPUT_FOUND, index=False)
missing_df.to_csv(OUTPUT_MISSING, index=False)


# =========================
# Summary
# =========================

print("Revisión terminada.")
print(f"Total de fechas en el CSV: {len(df)}")
print(f"Fechas con archivo encontrado: {len(found_df)}")
print(f"Fechas sin archivo encontrado: {len(missing_df)}")

print(f"\nReporte de archivos encontrados:")
print(OUTPUT_FOUND)

print(f"\nReporte de archivos faltantes:")
print(OUTPUT_MISSING)

if len(missing_df) > 0:
    print("\nPrimeros archivos faltantes:")
    print(missing_df.head(20).to_string(index=False))
else:
    print("\nNo faltan archivos.")