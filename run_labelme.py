import numpy as np
np.bool = bool  # Corrige o erro do NumPy >= 1.24

import runpy
import sys

if __name__ == "__main__":
    # Se quiser abrir um diretório específico, passe no argumento
    sys.argv = ["labelme", "dataset/raw_images"]
    runpy.run_module("labelme", run_name="__main__")
