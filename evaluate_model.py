from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt


class YoloEvaluator:
    def __init__(self, 
                 weights_path='results/pools_yolo/weights/best.pt',
                 data_yaml='dataset/yolo/data.yaml'):
        self.weights_path = Path(weights_path)
        self.data_yaml = Path(data_yaml)
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"❌ Modelo não encontrado em: {self.weights_path}")
        
        print(f"✅ Modelo carregado: {self.weights_path}")
        self.model = YOLO(self.weights_path)

    def evaluate(self):
        """
        Avalia o modelo YOLOv8 e retorna métricas
        """
        print("\n📊 AVALIAÇÃO DO MODELO YOLOv8\n")
        print(f"📁 Dataset: {self.data_yaml}")
        print("=" * 60)

        results = self.model.val(data=str(self.data_yaml))
        
        print("\n✅ Avaliação concluída!")
        print("=" * 60)
        print(f"📈 mAP50:  {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"📈 mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"📈 Precisão: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"📈 Recall:    {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
        print("=" * 60)

        return results

    def plot_metrics(self):
        """
        Mostra ou salva os gráficos de desempenho (curva PR e matriz de confusão)
        """
        import matplotlib
        matplotlib.use('Agg')  # evita erro com Qt/Wayland
        import matplotlib.pyplot as plt
        from pathlib import Path

        print("\n📉 Gerando gráficos de métricas...\n")

        # Tenta detectar automaticamente o diretório de avaliação mais recente
        detect_dir = Path("runs/detect")
        if detect_dir.exists():
            val_dirs = sorted(detect_dir.glob("val*"))
            if len(val_dirs) > 0:
                val_dir = val_dirs[-1]
            else:
                val_dir = detect_dir
        else:
            val_dir = self.weights_path.parent.parent / "val"

        pr_curve = val_dir / "PR_curve.png"
        conf_mat = val_dir / "confusion_matrix.png"

        if pr_curve.exists() or conf_mat.exists():
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            if pr_curve.exists():
                axs[0].imshow(plt.imread(pr_curve))
                axs[0].set_title("Curva Precisão-Recall")
                axs[0].axis("off")
            else:
                axs[0].text(0.5, 0.5, "PR_curve.png não encontrado", ha="center", va="center")

            if conf_mat.exists():
                axs[1].imshow(plt.imread(conf_mat))
                axs[1].set_title("Matriz de Confusão")
                axs[1].axis("off")
            else:
                axs[1].text(0.5, 0.5, "confusion_matrix.png não encontrado", ha="center", va="center")

            plt.tight_layout()

            # Salvar imagem combinada
            output_path = val_dir / "metrics_summary.png"
            plt.savefig(output_path)
            print(f"✅ Gráficos salvos em: {output_path}")
        else:
            print("⚠️ Nenhum gráfico encontrado em:", val_dir)



def main():
    evaluator = YoloEvaluator(
        weights_path='results/pools_yolo/weights/best.pt',
        data_yaml='dataset/yolo/data.yaml'
    )
    
    results = evaluator.evaluate()
    evaluator.plot_metrics()


if __name__ == "__main__":
    main()
