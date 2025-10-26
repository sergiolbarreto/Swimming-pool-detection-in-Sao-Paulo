import folium
from folium.plugins import HeatMap
import pandas as pd
from pathlib import Path

def main():
    print("\n🌎 GERANDO MAPA DE DENSIDADE DE PISCINAS (FOLIUM)...\n")

    # Caminhos de entrada
    detections_csv = Path("results/pools_sp_detection_full/detections_sp_full.csv")
    metadata_csv = Path("dataset/download_metadata.csv")
    output_html = Path("results/pools_sp_detection_full/pool_density_map.html")

    # Verificações
    if not detections_csv.exists():
        print(f"❌ Arquivo de detecções não encontrado: {detections_csv}")
        return
    if not metadata_csv.exists():
        print(f"❌ Arquivo de metadata não encontrado: {metadata_csv}")
        return

    # Leitura dos arquivos
    df_det = pd.read_csv(detections_csv)
    df_meta = pd.read_csv(metadata_csv)

    # Padronizar separadores e extrair nomes
    df_meta["filepath"] = df_meta["filepath"].astype(str).str.replace("\\", "/")
    df_meta["image"] = df_meta["filepath"].apply(lambda x: Path(x).name.strip())
    df_det["image"] = df_det["image"].apply(lambda x: Path(x).name.strip())

    # 🔹 Juntar as detecções com as coordenadas
    df_join = df_det.merge(df_meta, on="image", how="left")

    # Remover linhas sem coordenadas
    df_join = df_join.dropna(subset=["lat", "lon"])
    print(f"✅ {len(df_join)} detecções com coordenadas prontas para o mapa.\n")

    if len(df_join) == 0:
        print("⚠️ Nenhuma correspondência encontrada. Verifique se os nomes das imagens estão corretos.")
        print(df_det.head())
        print(df_meta.head())
        return

    # Criar mapa base centrado em São Paulo
    m = folium.Map(location=[-23.55, -46.63], zoom_start=10, tiles="cartodb positron")

    # Adicionar heatmap
    heat_data = [[row["lat"], row["lon"], 1] for _, row in df_join.iterrows()]
    HeatMap(
        heat_data,
        radius=10,
        blur=15,
        min_opacity=0.4,
        max_zoom=12
    ).add_to(m)

    # Adicionar marcador central
    folium.Marker(
        location=[-23.55, -46.63],
        popup="Centro de São Paulo",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

    # Salvar HTML
    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(output_html)

    print(f"✅ Mapa de densidade salvo em: {output_html}\n")
    print("Abra o arquivo no navegador para visualizar o mapa interativo.\n")

if __name__ == "__main__":
    main()
