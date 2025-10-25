import json
import shutil
from pathlib import Path
import random
from PIL import Image
import numpy as np

class DatasetPreparer:
    def __init__(self, 
                 images_dir='dataset/raw_images',
                 annotations_dir='dataset/annotations',
                 output_dir='dataset/yolo'):
        
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        
        # Criar estrutura YOLO
        self.splits = ['train', 'val', 'test']
        for split in self.splits:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def convert_labelme_to_yolo(self, json_path, img_width, img_height):
        """
        Converte anotações do labelme (JSON) para formato YOLO
        
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        Todas coordenadas normalizadas (0-1)
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        yolo_annotations = []
        
        for shape in data.get('shapes', []):
            if shape['shape_type'] == 'rectangle':
                points = shape['points']
                
                # Pontos do retângulo [top-left, bottom-right]
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # Calcular centro e dimensões
                x_center = (x1 + x2) / 2.0 / img_width
                y_center = (y1 + y2) / 2.0 / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height
                
                # Classe 0 = pool
                class_id = 0
                
                yolo_annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
            
            elif shape['shape_type'] == 'polygon':
                # Para polígonos, calcular bounding box
                points = shape['points']
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                
                x_center = (x1 + x2) / 2.0 / img_width
                y_center = (y1 + y2) / 2.0 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                class_id = 0
                
                yolo_annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
        
        return yolo_annotations
    
    def prepare_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        """
        Prepara dataset completo
        
        Args:
            train_ratio: Proporção para treino (0.7 = 70%)
            val_ratio: Proporção para validação (0.2 = 20%)
            test_ratio: Proporção para teste (0.1 = 10%)
        """
        random.seed(seed)
        
        # Listar todas as imagens
        image_files = list(self.images_dir.glob('*.jpg'))
        
        # Listar todas as anotações
        annotation_files = list(self.annotations_dir.glob('*.json'))
        annotation_stems = {f.stem for f in annotation_files}
        
        print(f"\n{'='*60}")
        print(f"PREPARAÇÃO DO DATASET YOLO")
        print(f"{'='*60}")
        print(f"Total de imagens: {len(image_files)}")
        print(f"Imagens com anotações: {len(annotation_files)}")
        print(f"Imagens sem piscinas: {len(image_files) - len(annotation_files)}")
        print(f"{'='*60}\n")
        
        # Embaralhar
        random.shuffle(image_files)
        
        # Calcular splits
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        splits_data = {
            'train': image_files[:n_train],
            'val': image_files[n_train:n_train+n_val],
            'test': image_files[n_train+n_val:]
        }
        
        print(f"📊 Divisão do dataset:")
        print(f"  Train: {n_train} imagens ({train_ratio*100:.0f}%)")
        print(f"  Val:   {n_val} imagens ({val_ratio*100:.0f}%)")
        print(f"  Test:  {n_test} imagens ({test_ratio*100:.0f}%)")
        print(f"{'='*60}\n")
        
        stats = {'train': {'with_pools': 0, 'without_pools': 0},
                'val': {'with_pools': 0, 'without_pools': 0},
                'test': {'with_pools': 0, 'without_pools': 0}}
        
        # Processar cada split
        for split_name, images in splits_data.items():
            print(f"Processando {split_name}...")
            
            for img_path in images:
                img_stem = img_path.stem
                
                # Copiar imagem
                dst_img = self.output_dir / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Processar anotação (se existir)
                json_path = self.annotations_dir / f"{img_stem}.json"
                
                if json_path.exists():
                    # Tem anotação - converter para YOLO
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    
                    yolo_annotations = self.convert_labelme_to_yolo(
                        json_path, img_width, img_height
                    )
                    
                    # Salvar arquivo .txt
                    txt_path = self.output_dir / split_name / 'labels' / f"{img_stem}.txt"
                    with open(txt_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    
                    stats[split_name]['with_pools'] += 1
                else:
                    # Sem piscinas - criar arquivo vazio
                    txt_path = self.output_dir / split_name / 'labels' / f"{img_stem}.txt"
                    txt_path.touch()
                    
                    stats[split_name]['without_pools'] += 1
            
            print(f"  ✅ {split_name}: {len(images)} imagens processadas")
        
        print(f"\n{'='*60}")
        print(f"✅ Dataset preparado com sucesso!")
        print(f"{'='*60}\n")
        
        # Estatísticas detalhadas
        print("📊 Distribuição de piscinas por split:")
        for split_name in self.splits:
            with_pools = stats[split_name]['with_pools']
            without_pools = stats[split_name]['without_pools']
            total = with_pools + without_pools
            
            print(f"\n{split_name.upper()}:")
            print(f"  Com piscinas:    {with_pools:2d} ({with_pools/total*100:.1f}%)")
            print(f"  Sem piscinas:    {without_pools:2d} ({without_pools/total*100:.1f}%)")
        
        return stats
    
    def create_data_yaml(self):
        """
        Cria arquivo data.yaml para YOLO
        """
        yaml_content = f"""# Dataset de Piscinas de São Paulo
path: {self.output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: pool

# Informações adicionais
nc: 1  # número de classes
"""
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✅ Arquivo data.yaml criado em: {yaml_path}")
        
        return yaml_path


def main():
    print("\n🏊 PREPARAÇÃO DO DATASET PARA YOLO\n")
    
    # Verificar se diretórios existem
    images_dir = Path('dataset/raw_images')
    annotations_dir = Path('dataset/annotations')
    
    if not images_dir.exists():
        print(f"❌ ERRO: Diretório {images_dir} não encontrado!")
        return
    
    if not annotations_dir.exists():
        print(f"❌ ERRO: Diretório {annotations_dir} não encontrado!")
        return
    
    # Criar preparador
    preparer = DatasetPreparer()
    
    # Preparar dataset
    stats = preparer.prepare_dataset(
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
    
    # Criar data.yaml
    yaml_path = preparer.create_data_yaml()



if __name__ == "__main__":
    main()