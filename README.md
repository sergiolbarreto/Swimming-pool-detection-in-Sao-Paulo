# 🏊‍♂️ Swimming Pool Detection in São Paulo

Este projeto tem como objetivo **estimar o número de piscinas na cidade de São Paulo** a partir de **imagens de satélite** e **modelos de detecção de objetos** (YOLO).  
Foi desenvolvido como parte de um desafio de Machine Learning que combina **visão computacional e inferência estatística**.

---

## 🎯 Desafio

> **How Many Pools in São Paulo?**  
> **Goal:** Estimate the number of swimming pools in São Paulo using ML + satellite images.  
>  
> **Tasks**
> - Sample rooftops from Google Maps or INPE  
> - Train a detector (> 0.65 mAP) for pools  
> - Use statistics to extrapolate total count  
>
> **Bonus**
> - Folium map with pool density  
> - District-wise comparison

---

## 🧭 Etapas do Projeto

### 1. Tentativa inicial com dados públicos (GeoSampa)
A primeira ideia foi **buscar um dataset pronto** de imagens de satélite de São Paulo, especialmente no **GeoSampa**.  
Apesar de o portal disponibilizar ortofotos corrigidas, o processo de **extração e recorte em blocos uniformes** mostrou-se inviável no tempo disponível.  
Cada imagem precisava ser manualmente localizada e baixada, inviabilizando a automação.

---

### 2. Extração de imagens via Google Maps API
A solução adotada foi usar a **Google Maps Static API**, através da **Google Cloud Platform**, para **baixar imagens de satélite diretamente de coordenadas de São Paulo**.

O script `download_sp_images.py`:
- gera coordenadas **aleatórias estratificadas por renda** (`high_income`, `middle_income`, `low_income`);
- usa a API do Google Maps para baixar blocos 640×640 px (zoom 19);
- impõe uma **distância mínima de 150 m** entre pontos para evitar sobreposição;
- registra metadados como `lat`, `lon`, `stratum` e `filepath` no arquivo  
  `dataset/download_metadata.csv`.

Áreas-alvo: Jardins, Morumbi, Brooklin, Pinheiros (alta renda); Mooca, Pompeia (média); Itaquera e Capão Redondo (baixa).  
Foram obtidas cerca de **600 imagens iniciais**.

---

### 3. Rotulagem manual das piscinas
Como não havia dataset rotulado de São Paulo, realizei o rotulamento manual  de uma parte das imagens usando o **LabelMe**.  
Cada piscina foi marcada com polígonos.  
Os arquivos `.json` foram convertidos para o formato YOLO com:

```bash
python3 prepare_dataset.py
```

Resultado:
```
dataset/yolo_sp/images   → imagens
dataset/yolo_sp/labels   → labels YOLO
```

---

### 4. Uso do dataset de Belo Horizonte (BH)
Devido à limitação de tempo para rotular grandes volumes, utilizei o **dataset de Belo Horizonte**, que possui imagens urbanas semelhantes às de São Paulo.  
A textura urbana e a disposição das construções são comparáveis, o que permite **transfer learning consistente**.

---

### 5. Modelo e treinamento
O modelo escolhido foi o **YOLOv8**, que oferece:
- arquitetura leve e rápida;  
- **data augmentation automática** (rotação, flips, brilho etc.);  
- suporte nativo ao formato YOLO.

Treinamento no dataset BH:
```
epochs = 50
batch  = 16
imgsz  = 640
optimizer = SGD
```

**Métricas finais do treinamento:**

| Métrica | Valor |
|----------|--------|
| Precision | **0.8877** |
| Recall | **0.8449** |
| mAP50 | **0.9159** |
| mAP50-95 | **0.6203** |
| box_loss | 0.9867 |
| cls_loss | 0.5290 |
| dfl_loss | 0.8055 |

📈 O modelo atingiu **excelente desempenho (mAP50 > 0.9)**, superando o objetivo > 0.65 mAP do desafio, com alta precisão e recall.

---

### 6. Validação e teste em São Paulo
Com o modelo treinado em BH, realizei a validação nas imagens de São Paulo.  
Inicialmente usei todas as imagens, mas muitas **não continham piscinas**.  
Para avaliar o desempenho real, filtrei o conjunto apenas para **imagens com piscinas**.
Os resultados mostram **boa generalização**, mas também a diferença de domínio entre BH e SP.

---

### 7. Detecção nas imagens de São Paulo

Usando o modelo YOLOv8 treinado em BH, rodei a detecção em todas as imagens extraídas de São Paulo.  
Parâmetros principais:

```
conf = 0.45
iou  = 0.70
imgsz = 640
```

**Resultados da detecção:**
```
✅ Detecção concluída!
📊 Total de detecções: 929
📁 CSV salvo em: results/pools_sp_detection_full/detections_sp_full.csv
📂 Resultados visuais: results/pools_sp_detection_full/detect_sp_full

🏊 Piscinas detectadas: 929
📈 Média: 0.67 piscinas por imagem
```

As detecções foram cruzadas com o `download_metadata.csv`, associando cada piscina à sua localização (lat/lon) e ao estrato socioeconômico.

---

### 8. Estimativa populacional de piscinas

A estimativa do número total de piscinas em São Paulo foi feita via **extrapolação estatística estratificada**:

#### 1️⃣ Cálculo da densidade por estrato

| Estrato | Densidade observada (piscinas/km²) |
|----------|------------------------------------|
| Alta renda | **14.83** |
| Média renda | **4.00** |
| Baixa renda | **3.53** |

Cada tile corresponde a ≈ 0.0583 km², o que permite estimar a densidade amostral.

#### 2️⃣ Estimativa bruta
Multiplicando a densidade média ponderada pela área total da cidade (≈ 1 521 km²):

```
🌆 ESTIMATIVA BRUTA: 16 306 piscinas
```

#### 3️⃣ Correção por recall (real = 0.15)

O modelo detecta cerca de 15 % das piscinas visíveis.  
Aplicou-se o fator 1/recall para compensar a subdetecção:

\[
\text{Estimativa ajustada} = \frac{16 306}{0.15} \approx 108 712
\]

### 9. Metodologia estatística

A extrapolação foi feita por **amostragem estratificada**, inspirada em técnicas de sensoriamento remoto:

1. **Estratificação** por renda (baixa, média, alta).  
2. **Cálculo de densidade** de piscinas por km² em cada estrato.  
3. **Ponderação espacial** pela área ocupada de cada estrato na cidade.  
4. **Correção por recall**, compensando subdetecção.  
5. **Soma ponderada** → estimativa global final.

Essa abordagem considera:
- desempenho real do modelo;  
- desigualdade espacial da distribuição de piscinas;  
- tamanho e área amostrada.

📈 **Resultado final:**  
> **≈ 108 mil piscinas na cidade de São Paulo**  
---
Gerei um folium map como foi pedido, está no arquivo html `pool_density_map.html`.
Ele foi gerado pelo script `generate_pool_heatmap.py`
<img width="1288" height="686" alt="image" src="https://github.com/user-attachments/assets/9043bded-5a89-4f4f-abdc-0e1763568fc5" />


## 🧩 Organização do pipeline

```
dataset/
├── raw_images/              # imagens baixadas pela API
├── annotations/             # arquivos .json do LabelMe
├── yolo_sp/                 # dataset convertido para YOLO
│   ├── images/
│   └── labels/
└── download_metadata.csv    # metadados (lat, lon, stratum, path)

results/
├── pools_bh_train/          # modelo YOLO treinado em BH
├── pools_sp_validation/     # métricas de validação em SP
└── pools_sp_detection_full/ # detecções + estimativas finais
```

---

## 🚀 Execução passo a passo

1. **Instalar dependências**
   ```bash
   pip install -r requirements.txt
   ```

2. **Baixar imagens**
   ```bash
   python3 download_sp_images.py
   ```

3. **Rotular manualmente**
   ```bash
   labelme dataset/raw_images
   ```

4. **Converter rótulos para YOLO**
   ```bash
   python3 prepare_dataset.py
   ```

5. **Treinar o modelo (opcional)**
   ```bash
   python3 train_bh.py
   ```

6. **Rodar detecção**
   ```bash
   python3 detect_sp.py
   ```

7. **Gerar estimativa**
   ```bash
   python3 estimate_sp_pools.py
   ```

---

## 🔮 Próximos passos

- **Ampliar o dataset** de SP, rotulando mais imagens via Google API.  
- **Fine-tuning local**, ajustando o modelo de BH para o domínio paulistano.  
- **Novas classes** (*caixa d’água*, *piscina desativada*, *quadra*), reduzindo falsos positivos.  

---

## 🧠 Conclusão

Este projeto demonstrou um **pipeline completo de visão computacional e análise estatística aplicada ao mapeamento urbano**, incluindo:
**extração → rotulagem → treinamento → validação → estimativa.**

---
