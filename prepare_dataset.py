import json
import shutil
from pathlib import Path
from PIL import Image


class DatasetPreparer:
    def __init__(self,
                 images_dir='dataset/yolo_sp/images',
                 annotations_dir='dataset/annotations',
                 output_labels_dir='dataset/yolo_sp/labels'):

        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_labels_dir = Path(output_labels_dir)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)

    def convert_labelme_to_yolo(self, json_path, img_width, img_height):
        """
        Converte anotações do LabelMe (JSON) para formato YOLO.
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        shapes = data.get('shapes', [])
        if not shapes:
            return []  # sem anotações

        yolo_annotations = []
        for shape in shapes:
            if 'points' not in shape:
                continue

            points = shape['points']
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            x_center = (x1 + x2) / 2.0 / img_width
            y_center = (y1 + y2) / 2.0 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            class_id = 0  # 0 = piscina
            yolo_annotations.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        return yolo_annotations

    def prepare_dataset(self):
        print("\n🏊 PREPARANDO DATASET PARA YOLO\n")
        print("=" * 60)

        json_files = list(self.annotations_dir.glob("*.json"))
        print(f"📄 Total de anotações LabelMe encontradas: {len(json_files)}")

        converted = 0
        skipped = 0

        for json_path in json_files:
            stem = json_path.stem
            img_path = self.images_dir / f"{stem}.jpg"

            if not img_path.exists():
                print(f"⚠️ Imagem não encontrada para {stem}, pulando.")
                skipped += 1
                continue

            try:
                img = Image.open(img_path)
                w, h = img.size
            except Exception as e:
                print(f"❌ Erro ao abrir imagem {img_path}: {e}")
                skipped += 1
                continue

            yolo_annotations = self.convert_labelme_to_yolo(json_path, w, h)
            txt_path = self.output_labels_dir / f"{stem}.txt"

            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_annotations))

            converted += 1

        print(f"\n✅ Conversão concluída!")
        print(f"📁 Labels YOLO salvos em: {self.output_labels_dir}")
        print(f"🗺️ Arquivos processados: {converted}")
        print(f"⚠️ Ignorados (sem imagem ou erro): {skipped}")
        print("=" * 60)

        # Criar YAML
        yaml_path = Path(self.output_labels_dir).parent / "data.yaml"
        yaml_content = f"""# Dataset São Paulo Unificado
train:
val: images
test:
names:
  0: pool
"""
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"✅ Arquivo data.yaml criado em: {yaml_path}")
        return converted, skipped


if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.prepare_dataset()
