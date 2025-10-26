# detect_sp_all_cpu_safe.py
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import torch
from itertools import chain
from tqdm import tqdm

def main():
    print("\n🏙️ DETECÇÃO DE PISCINAS NAS IMAGENS DE SÃO PAULO (VAL + TEST)\n")

    # Caminhos
    model_path = Path("results/pools_bh_train/weights/best.pt")
    val_dir = Path("dataset/yolo_sp/val/images")
    test_dir = Path("dataset/yolo_sp/test/images")
    output_dir = Path("results/pools_sp_detection_all")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combinar imagens de validação e teste
    all_images = list(chain(val_dir.glob("*.jpg"), test_dir.glob("*.jpg")))
    print(f"📷 Total de imagens a processar: {len(all_images)}")

    # Detectar GPU ou CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Dispositivo detectado: {device.upper()}")

    # Carregar modelo
    model = YOLO(model_path)

    # 🔹 Rodar em lotes pequenos para evitar travamentos
    batch_size = 2
    all_detections = []

    print("\n🚀 Iniciando inferência em lotes pequenos...\n")
    for i in tqdm(range(0, len(all_images), batch_size), desc="Processando"):
        batch = all_images[i:i + batch_size]

        results = model.predict(
            source=[str(p) for p in batch],
            conf=0.45,
            iou=0.7,
            imgsz=640,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(output_dir),
            name="detect_sp_all",
            exist_ok=True,
            device=device,
            verbose=False
        )

        # 🔹 Salvar resultados parciais em memória
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xywh.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                names = [model.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
                for i in range(len(boxes)):
                    all_detections.append({
                        "image": Path(r.path).name,
                        "class": names[i],
                        "confidence": float(confs[i]),
                        "x_center": float(boxes[i][0]),
                        "y_center": float(boxes[i][1]),
                        "width": float(boxes[i][2]),
                        "height": float(boxes[i][3])
                    })

    # 🔹 Salvar CSV consolidado
    df = pd.DataFrame(all_detections)
    csv_path = output_dir / "detections_sp_all.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Detecção concluída!")
    print(f"📊 Total de detecções: {len(df)}")
    print(f"📁 CSV salvo em: {csv_path}")
    print(f"📂 Resultados visuais: {output_dir / 'detect_sp_all'}")

    total_pools = len(df)
    print(f"\n🏊 Piscinas detectadas: {total_pools}")
    print(f"📈 Média: {total_pools / len(all_images):.2f} por imagem")

if __name__ == "__main__":
    main()
