import pandas as pd
from pathlib import Path

def main():
    print("\n📊 ESTIMATIVA POPULACIONAL DE PISCINAS EM SÃO PAULO\n")

    det_path = Path("results/pools_sp_detection_full/detections_sp_full.csv")
    meta_path = Path("dataset/download_metadata.csv")
    output_dir = det_path.parent

    df_det = pd.read_csv(det_path)
    df_meta = pd.read_csv(meta_path)

    df_det["cell_id"] = df_det["image"].str.extract(r"(\d+)")
    df_meta["cell_id"] = df_meta["filepath"].str.extract(r"(\d+)")
    df = df_det.merge(df_meta, on="cell_id", how="left").dropna(subset=["lat", "lon", "stratum"])

    # Densidades reais por estrato
    area_tile = 0.0583
    df_area = df_meta.groupby("stratum")["filepath"].count().reset_index()
    df_area["area_km2"] = df_area["filepath"] * area_tile
    df_pools = df.groupby("stratum")["image"].count().reset_index(name="pools")

    df_est = df_area.merge(df_pools, on="stratum", how="left").fillna(0)
    df_est["density"] = df_est["pools"] / df_est["area_km2"]

    print("\n🏙️ Densidades observadas (piscinas/km²):")
    print(df_est[["stratum", "density"]])

    # --- Correção e ponderação ---
    area_sp_total = 1521  # km² (área da cidade de São Paulo)
    area_share = {"high_income": 0.25, "middle_income": 0.50, "low_income": 0.25}
    correction = {"high_income": 2.0, "middle_income": 1.3, "low_income": 0.8}

    df_est["area_share"] = df_est["stratum"].map(area_share)
    df_est["corr_factor"] = df_est["stratum"].map(correction)
    df_est["area_sp_km2"] = df_est["area_share"] * area_sp_total
    df_est["density_corr"] = df_est["density"] * df_est["corr_factor"]
    df_est["pools_est"] = df_est["density_corr"] * df_est["area_sp_km2"]

    total_est = df_est["pools_est"].sum()

    # --- Ajuste leve por recall real ---
    recall_real = 0.15
    total_est_corr = total_est / recall_real

    print(f"\n🌆 ESTIMATIVA BRUTA (sem correção de recall): {int(total_est):,} piscinas")
    print(f"🔧 Ajuste aplicado por recall real = {recall_real:.4f}")
    print(f"🌇 ESTIMATIVA FINAL AJUSTADA: {int(total_est_corr):,} piscinas")

    # --- Salvar ---
    out_csv = output_dir / "estimativa_por_estrato_ajustada.csv"
    df_est.to_csv(out_csv, index=False)
    out_txt = output_dir / "estimativa_geral_por_estrato.txt"
    with open(out_txt, "w") as f:
        f.write(df_est.to_string(index=False))
        f.write(f"\n\nEstimativa bruta: {int(total_est):,}")
        f.write(f"\nEstimativa ajustada (recall={recall_real:.4f}): {int(total_est_corr):,}\n")

    print(f"\n✅ Resultados salvos em: {out_csv}")

if __name__ == "__main__":
    main()
