from ultralytics import YOLO
from pathlib import Path

class YOLOInference:
    def __init__(self, 
                 weights_path='results/pools_yolo/weights/best.pt',
                 input_dir='dataset/test_images',
                 output_dir='runs/predict',
                 conf=0.25):
        
        self.weights_path = Path(weights_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.conf = conf

        if not self.weights_path.exists():
            raise FileNotFoundError(f"❌ Modelo não encontrado em: {self.weights_path}")
        if not self.input_dir.exists():
            raise FileNotFoundError(f"❌ Diretório de entrada não encontrado: {self.input_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ Modelo carregado: {self.weights_path}")
        print(f"📂 Imagens de entrada: {self.input_dir}")
        print(f"💾 Resultados serão salvos em: {self.output_dir}\n")
        
        self.model = YOLO(self.weights_path)

    def run(self):
        """
        Executa inferência sobre todas as imagens do diretório
        """
        images = list(self.input_dir.glob('*.jpg')) + list(self.input_dir.glob('*.png'))
        if len(images) == 0:
            print("⚠️ Nenhuma imagem encontrada no diretório de entrada.")
            return

        print(f"🔍 Rodando inferência em {len(images)} imagem(ns)...\n")

        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] → {img_path.name}")
            self.model.predict(
                source=str(img_path),
                conf=self.conf,
                save=True,
                project=str(self.output_dir),
                name='pools_inference',
                exist_ok=True
            )

        print("\n✅ Inferência concluída com sucesso!")
        print(f"📸 Resultados disponíveis em: {self.output_dir / 'pools_inference'}")

    def infer_single(self, image_path):
        """
        Executa inferência em uma única imagem
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"❌ Imagem não encontrada: {image_path}")
        
        print(f"\n🔎 Rodando inferência em: {image_path.name}")
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            save=True,
            project=str(self.output_dir),
            name='single_image',
            exist_ok=True
        )

        print("✅ Inferência concluída!")
        print(f"📸 Resultado salvo em: {self.output_dir / 'single_image'}")
        return results


def main():
    inference = YOLOInference(
        weights_path='results/pools_yolo/weights/best.pt',
        input_dir='dataset/yolo/test/images',  # usa o split de teste gerado no prepare_dataset.py
        output_dir='runs/predict',
        conf=0.25
    )
    
    inference.run()
    # Exemplo opcional:
    # inference.infer_single('dataset/yolo/test/images/cell_0021.jpg')


if __name__ == "__main__":
    main()
