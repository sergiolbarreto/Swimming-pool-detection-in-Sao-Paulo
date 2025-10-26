from pathlib import Path
import shutil
import yaml

def main():
    labels_dir = Path("dataset/yolo_sp/labels")
    images_dir = Path("dataset/yolo_sp/images")
    out_images_dir = Path("dataset/yolo_sp_with_pools/images")
    out_labels_dir = Path("dataset/yolo_sp_with_pools/labels")
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    pool_labels = [f for f in labels_dir.glob("*.txt") if f.stat().st_size > 0]
    print(f"🏊 {len(pool_labels)} imagens com piscina identificadas.")

    for lbl_path in pool_labels:
        img_name = lbl_path.stem + ".jpg"
        img_path = images_dir / img_name
        if img_path.exists():
            shutil.copy2(img_path, out_images_dir / img_name)
            shutil.copy2(lbl_path, out_labels_dir / lbl_path.name)

    # cria data.yaml
    data = {
        "train": "",
        "val": str(out_images_dir),
        "test": "",
        "names": {0: "pool"}
    }

    yaml_path = Path("dataset/yolo_sp_with_pools/data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    print(f"✅ Dataset criado em: {out_images_dir.parent}")
    print(f"🗂️ Arquivo data.yaml salvo em: {yaml_path}")

if __name__ == "__main__":
    main()
