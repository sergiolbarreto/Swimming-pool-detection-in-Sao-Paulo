# detect_sp.py
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

def main():
    print("\n🏙️ DETECÇÃO DE PISCINAS NAS IMAGENS DE SÃO PAULO\n")

    # Caminhos principais
    model_path = Path("results/pools_bh_train/weights/best.pt")
    images_dir = Path("dataset/raw_images")
    output_dir = Path("results/pools_sp_detection_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Listar imagens
    image_files = sorted(list(images_dir.glob("*.jpg")))
    print(f"📷 Total de imagens a processar: {len(image_files)}")

    # Detectar GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Dispositivo detectado: {device.upper()}")

    # Carregar modelo YOLO
    model = YOLO(model_path)

    # Configurações de detecção
    conf_threshold = 0.45
    iou_threshold = 0.7
    batch_size = 4  # Pequenos lotes para evitar travamentos

    all_detections = []

    print("\n🚀 Iniciando inferência...\n")

    for i in tqdm(range(0, len(image_files), batch_size), desc="Processando"):
        batch = image_files[i:i + batch_size]

        results = model.predict(
            source=[str(p) for p in batch],
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(output_dir),
            name="detect_sp_full",
            exist_ok=True,
            device=device,
            verbose=False
        )

        # Extrair detecções do batch
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xywh.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for j in range(len(boxes)):
                    all_detections.append({
                        "image": Path(r.path).name,
                        "class": model.names[clss[j]],
                        "confidence": float(confs[j]),
                        "x_center": float(boxes[j][0]),
                        "y_center": float(boxes[j][1]),
                        "width": float(boxes[j][2]),
                        "height": float(boxes[j][3]),
                    })

    # Salvar CSV consolidado
    df = pd.DataFrame(all_detections)
    csv_path = output_dir / "detections_sp_full.csv"
    df.to_csv(csv_path, index=False)

    # Resumo
    print(f"\n✅ Detecção concluída!")
    print(f"📊 Total de detecções: {len(df)}")
    print(f"📁 CSV salvo em: {csv_path}")
    print(f"📂 Resultados visuais: {output_dir / 'detect_sp_full'}")

    total_images = len(image_files)
    avg_pools = len(df) / total_images if total_images > 0 else 0
    print(f"\n🏊 Piscinas detectadas: {len(df)}")
    print(f"📈 Média: {avg_pools:.2f} piscinas por imagem")

if __name__ == "__main__":
    main()
