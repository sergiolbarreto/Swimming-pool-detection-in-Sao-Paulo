# download_sp_all_strata.py
import pandas as pd
import random
import requests
from pathlib import Path
import math
import time
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# ---------------------------------------------------------
API_KEY = ""  # coloque aqui a sua api key do google
N_SAMPLES_PER_STRATUM = 300
MIN_DIST_METERS = 150
OUTPUT_DIR = Path("dataset/raw_images")
METADATA_PATH = Path("dataset/download_metadata.csv")

# ---------------------------------------------------------
# REGIÕES-ALVO POR ESTRATO
# ---------------------------------------------------------
AREAS = {
    "high_income": [
        (-23.587, -46.676),  # Jardins
        (-23.601, -46.722),  # Morumbi
        (-23.604, -46.658),  # Vila Olímpia
        (-23.625, -46.698),  # Brooklin
        (-23.556, -46.665),  # Pinheiros
        (-23.571, -46.680),  # Alto de Pinheiros
    ],
    "middle_income": [
        (-23.543, -46.616),  # Vila Mariana
        (-23.521, -46.595),  # Mooca
        (-23.512, -46.634),  # Pompeia
        (-23.497, -46.675),  # Lapa
        (-23.533, -46.655),  # Água Branca
    ],
    "low_income": [
        (-23.681, -46.602),  # Capão Redondo
        (-23.688, -46.713),  # Jardim Ângela
        (-23.698, -46.726),  # Grajaú
        (-23.705, -46.608),  # Cidade Ademar
        (-23.651, -46.539),  # Itaquera
    ]
}

# ---------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def already_exists(lat, lon, df):
    for _, row in df.iterrows():
        if haversine(lat, lon, row["lat"], row["lon"]) < MIN_DIST_METERS:
            return True
    return False

def download_google_image(lat, lon, cell_id):
    """Baixa uma imagem de satélite centrada em (lat, lon)"""
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom=19&size=640x640&maptype=satellite&key={API_KEY}"
    )
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            out_path = OUTPUT_DIR / f"cell_{cell_id}.jpg"
            out_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False

# ---------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if METADATA_PATH.exists():
        df_meta = pd.read_csv(METADATA_PATH)
    else:
        df_meta = pd.DataFrame(columns=["cell_id", "lat", "lon", "stratum", "filepath", "success"])

    new_records = []

    print("\n🌎 Iniciando download estratificado de imagens de São Paulo...\n")

    for stratum, region_list in AREAS.items():
        print(f"🗺️ Estrato: {stratum} — Gerando {N_SAMPLES_PER_STRATUM} amostras")

        for _ in tqdm(range(N_SAMPLES_PER_STRATUM), desc=f"Baixando {stratum}"):
            base_lat, base_lon = random.choice(region_list)
            lat = base_lat + random.uniform(-0.02, 0.02)
            lon = base_lon + random.uniform(-0.02, 0.02)

            if already_exists(lat, lon, df_meta):
                continue

            cell_id = int(time.time() * 1000) % 10_000_000
            img_path = OUTPUT_DIR / f"cell_{cell_id}.jpg"

            success = download_google_image(lat, lon, cell_id)

            if success:
                new_records.append({
                    "cell_id": cell_id,
                    "lat": lat,
                    "lon": lon,
                    "stratum": stratum,
                    "filepath": str(img_path),
                    "success": True
                })

            time.sleep(0.4)

    # -----------------------------------------------------
    # Salvar resultados
    # -----------------------------------------------------
    df_new = pd.DataFrame(new_records)
    df_final = pd.concat([df_meta, df_new], ignore_index=True)
    df_final.to_csv(METADATA_PATH, index=False)

    success_count = df_new["success"].sum()
    print(f"\n✅ {success_count} novas imagens baixadas.")
    print(f"📁 Total no metadata: {len(df_final)}")
    print(f"📂 Arquivo salvo em: {METADATA_PATH}\n")


if __name__ == "__main__":
    main()
