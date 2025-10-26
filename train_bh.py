"""
Treinamento YOLOv8 - Detecção de Piscinas (Dataset BH)

Executa:
1. Treinamento com ultralytics.YOLO
2. Salva logs e métricas em CSV
3. Exibe resumo final
"""

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import datetime

# =======================
# CONFIGURAÇÕES
# =======================
DATA_YAML = "dataset/yolo_bh/data.yaml"  # Dataset de treino
MODEL_NAME = "yolov8s.pt"
EPOCHS = 50
BATCH = 16
IMGSZ = 640
PROJECT = "results"
EXPERIMENT_NAME = "pools_bh_train"

# =======================
# EXECUÇÃO DO TREINAMENTO
# =======================
print("\n🏊 Iniciando treinamento YOLOv8 - Dataset Belo Horizonte\n")

# Carrega modelo base
model = YOLO(MODEL_NAME)

RUN_TRAINING = False  # <- coloque True se quiser treinar o modelo, se quiser vê só as métricas do modelo já treinado, deixe False


if RUN_TRAINING:
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT,
        name=EXPERIMENT_NAME,
        exist_ok=True
    )
else:
    print("⚠️ Treinamento desativado (RUN_TRAINING=False).")


# =======================
# EXPORTAÇÃO DAS MÉTRICAS
# =======================
print("\n📊 Salvando métricas do treinamento...")

# Diretório do experimento
exp_dir = Path(PROJECT) / EXPERIMENT_NAME
metrics_file = exp_dir / "results.csv"
summary_file = exp_dir / "summary.txt"

if metrics_file.exists():
    df = pd.read_csv(metrics_file)

    # Extrai última linha (última época)
    last = df.iloc[-1]

    summary = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": EPOCHS,
        "batch": BATCH,
        "imgsz": IMGSZ,
        "mAP50": last.get("metrics/mAP50(B)", None),
        "mAP50-95": last.get("metrics/mAP50-95(B)", None),
        "precision": last.get("metrics/precision(B)", None),
        "recall": last.get("metrics/recall(B)", None),
        "box_loss": last.get("train/box_loss", None),
        "cls_loss": last.get("train/cls_loss", None),
        "dfl_loss": last.get("train/dfl_loss", None),
    }

    # Salva resumo em .txt
    with open(summary_file, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"✅ Métricas salvas em: {metrics_file}")
    print(f"✅ Resumo salvo em: {summary_file}")

    print("\n📈 RESULTADOS FINAIS:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

else:
    print("⚠️ Não foi possível encontrar 'results.csv' — o YOLO pode não ter salvo métricas.")
    print("Verifique o diretório:", exp_dir)

print("\n✅ Treinamento concluído com sucesso!\n")
