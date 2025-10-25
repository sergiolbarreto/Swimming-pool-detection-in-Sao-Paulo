import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

# Caminhos
root = Path("BH-DATASET/BH-POOLS")
output = Path("dataset/yolo_bh")
(output / "images").mkdir(parents=True, exist_ok=True)
(output / "labels").mkdir(parents=True, exist_ok=True)

total_images = 0
total_pools = 0

for region in sorted(root.glob("REGION_*")):
    img_dir = region / "IMAGES"
    mask_dir = region / "ANNOTATION"

    print(f"\n📂 Processando {region.name} ...")

    region_images = 0
    region_pools = 0

    for img_path in tqdm(img_dir.glob("*.jpg")):
        mask_path = mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            continue

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), 0)
        h, w = mask.shape[:2]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            x_c = (x + bw/2) / w
            y_c = (y + bh/2) / h
            bw_n = bw / w
            bh_n = bh / h
            lines.append(f"0 {x_c:.6f} {y_c:.6f} {bw_n:.6f} {bh_n:.6f}")

        # Novo nome único por região
        new_name = f"{region.name}_{img_path.stem}.jpg"
        new_label = f"{region.name}_{img_path.stem}.txt"

        shutil.copy2(img_path, output / "images" / new_name)
        with open(output / "labels" / new_label, "w") as f:
            f.write("\n".join(lines))

        region_images += 1
        if len(lines) > 0:
            region_pools += 1

    total_images += region_images
    total_pools += region_pools
    print(f"✅ {region.name}: {region_images} imagens | {region_pools} com piscinas detectadas")

print(f"\n🏁 Concluído! Total: {total_images} imagens | {total_pools} com piscinas")
