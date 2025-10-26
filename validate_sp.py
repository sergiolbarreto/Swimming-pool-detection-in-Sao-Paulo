from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import torch

def main():
    print("\n🏙️ VALIDANDO MODELO (Treinado em BH) NAS IMAGENS DE SÃO PAULO\n")

    # Caminhos
    model_path = Path("results/pools_bh_train/weights/best.pt")
    data_yaml = Path("dataset/yolo_sp_with_pools/data.yaml")
    output_dir = Path("results/pools_sp_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verificar GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Dispositivo detectado: {device.upper()}")

    # Carregar modelo
    model = YOLO(model_path)

    # Validação
    results = model.val(
        data=str(data_yaml),
        imgsz=640,
        batch=16,
        project=str(output_dir),
        name="validate_sp",
        save_json=True,
        save_hybrid=True,
        save_conf=True,
        conf=0.45,
        iou=0.7,
        device=device
    )

    # Salvar métricas detalhadas
    metrics = results.results_dict
    metrics_path = output_dir / "metrics_sp.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("\n📊 MÉTRICAS DE VALIDAÇÃO (São Paulo):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n✅ Resultados salvos em: {output_dir}/")
    print("📈 Imagens com predições estão disponíveis em:")
    print(f"   {output_dir}/validate_sp/predictions/")

if __name__ == "__main__":
    main()
