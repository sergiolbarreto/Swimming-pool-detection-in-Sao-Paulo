import json
import shutil
from pathlib import Path
import random
from PIL import Image

class DatasetPreparer:
    def __init__(self, 
                 images_dir='dataset/raw_images',
                 annotations_dir='dataset/annotations',
                 output_dir='dataset/yolo_sp'):
        
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        
        # Agora só há val e test
        self.splits = ['val', 'test']
        for split in self.splits:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def convert_labelme_to_yolo(self, json_path, img_width, img_height):
        """Converte anotações do LabelMe (JSON) para formato YOLO"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        yolo_annotations = []
        for shape in data.get('shapes', []):
            points = shape['points']
            if shape['shape_type'] == 'rectangle':
                x1, y1 = points[0]
                x2, y2 = points[1]
            else:  # polygon → bounding box
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

            # Converter para formato YOLO (normalizado)
            x_center = (x1 + x2) / 2.0 / img_width
            y_center = (y1 + y2) / 2.0 / img_height
            width = abs(x2 - x1) / img_width
            height = abs(y2 - y1) / img_height
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations, len(yolo_annotations)
    
    def prepare_dataset(self, val_ratio=0.3, test_ratio=0.7, seed=42):
        """Cria dataset apenas com validação e teste (sem treino)"""
        random.seed(seed)
        
        image_files = list(self.images_dir.glob('*.jpg'))
        annotation_files = list(self.annotations_dir.glob('*.json'))
        annotation_stems = {f.stem for f in annotation_files}
        
        print(f"\n{'='*60}")
        print(f"PREPARAÇÃO DO DATASET YOLO (VALIDAÇÃO + TESTE)")
        print(f"{'='*60}")
        print(f"Total de imagens: {len(image_files)}")
        print(f"Imagens com anotações: {len(annotation_files)}")
        print(f"Imagens sem piscinas: {len(image_files) - len(annotation_files)}")
        print(f"{'='*60}\n")
        
        random.shuffle(image_files)
        n_total = len(image_files)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_val
        
        splits_data = {
            'val': image_files[:n_val],
            'test': image_files[n_val:]
        }
        
        print(f"📊 Divisão do dataset:")
        print(f"  Val:  {n_val} imagens ({val_ratio*100:.0f}%)")
        print(f"  Test: {n_test} imagens ({test_ratio*100:.0f}%)")
        print(f"{'='*60}\n")
        
        stats = {
            'val': {'with_pools': 0, 'without_pools': 0, 'total_pools': 0},
            'test': {'with_pools': 0, 'without_pools': 0, 'total_pools': 0}
        }
        
        for split_name, images in splits_data.items():
            print(f"Processando {split_name}...")
            for img_path in images:
                img_stem = img_path.stem
                dst_img = self.output_dir / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dst_img)

                json_path = self.annotations_dir / f"{img_stem}.json"
                txt_path = self.output_dir / split_name / 'labels' / f"{img_stem}.txt"

                if json_path.exists():
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    yolo_annotations, num_pools = self.convert_labelme_to_yolo(json_path, img_width, img_height)
                    with open(txt_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    stats[split_name]['with_pools'] += 1
                    stats[split_name]['total_pools'] += num_pools
                else:
                    txt_path.touch()
                    stats[split_name]['without_pools'] += 1
            
            print(f"  ✅ {split_name}: {len(images)} imagens processadas")
        
        print(f"\n{'='*60}")
        print(f"✅ Dataset preparado com sucesso!")
        print(f"{'='*60}\n")
        
        print("📊 Distribuição de piscinas por split:")
        for split_name in self.splits:
            with_pools = stats[split_name]['with_pools']
            without_pools = stats[split_name]['without_pools']
            total_pools = stats[split_name]['total_pools']
            total = with_pools + without_pools
            if total > 0:
                print(f"\n{split_name.upper()}:")
                print(f"  Imagens com piscinas: {with_pools:3d} ({with_pools/total*100:.1f}%)")
                print(f"  Imagens sem piscinas: {without_pools:3d} ({without_pools/total*100:.1f}%)")
                print(f"  Total de piscinas anotadas: {total_pools:3d}")
        
        return stats
    
    def create_data_yaml(self):
        """Cria arquivo data.yaml (sem split de treino)"""
        yaml_content = f"""# Dataset de Piscinas de São Paulo
path: {self.output_dir.absolute()}
val: val/images
test: test/images

names:
  0: pool

nc: 1
"""
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✅ Arquivo data.yaml criado em: {yaml_path}")
        return yaml_path


def main():
    print("\n🏊 PREPARAÇÃO DO DATASET DE VALIDAÇÃO E TESTE PARA YOLO\n")
    
    images_dir = Path('dataset/raw_images')
    annotations_dir = Path('dataset/annotations')
    
    if not images_dir.exists():
        print(f"❌ ERRO: Diretório {images_dir} não encontrado!")
        return
    if not annotations_dir.exists():
        print(f"❌ ERRO: Diretório {annotations_dir} não encontrado!")
        return
    
    preparer = DatasetPreparer(images_dir=images_dir, annotations_dir=annotations_dir)
    preparer.prepare_dataset(val_ratio=0.3, test_ratio=0.7, seed=42)
    preparer.create_data_yaml()


if __name__ == "__main__":
    main()
