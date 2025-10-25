import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # evita erro do OpenMP

from ultralytics import YOLO
from pathlib import Path

class YOLOTrainer:
    def __init__(self, 
                 data_yaml='dataset/yolo_bh/data.yaml',  # dataset BH
                 model_name='yolov8s.pt',
                 output_dir='results',
                 project_name='bh_pools_yolo',
                 epochs=100,
                 imgsz=640,
                 batch=16):
        
        self.data_yaml = Path(data_yaml)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.project_name = project_name
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        
        # Cria diretório de resultados
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        print("\n🚀 INICIANDO TREINAMENTO YOLOv8\n")
        print(f"📁 Dataset: {self.data_yaml}")
        print(f"🧠 Modelo base: {self.model_name}")
        print(f"📦 Resultados em: {self.output_dir}/{self.project_name}")
        print("="*60)

        model = YOLO(self.model_name)
        
        results = model.train(
            data=str(self.data_yaml),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            project=str(self.output_dir),
            name=self.project_name,
            exist_ok=True
        )

        print("\n✅ Treinamento concluído!")
        print(f"📊 Resultados salvos em: {self.output_dir / self.project_name}\n")
        return results

def main():
    trainer = YOLOTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
