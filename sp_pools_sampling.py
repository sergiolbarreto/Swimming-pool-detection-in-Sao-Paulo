"""
Pipeline para Amostragem e Coleta de Imagens de Satélite de São Paulo
Para detecção de piscinas usando ML

Requisitos:
pip install geopandas shapely folium pillow requests pandas numpy
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import requests
from pathlib import Path
import time
import json
from typing import List, Tuple
import folium

class SaoPauloSampler:
    """
    Classe para criar grid de amostragem estratificada de São Paulo
    """
    
    def __init__(self):
        # Bounding box aproximado de São Paulo (lat, lon)
        self.sp_bounds = {
            'lat_min': -23.8,
            'lat_max': -23.4,
            'lon_min': -46.85,
            'lon_max': -46.35
        }
        
        # Estratos baseados em características socioeconômicas
        self.strata = {
            'high_income': {
                'zones': ['Morumbi', 'Jardins', 'Pinheiros', 'Vila Mariana'],
                'expected_density': 'high',
                'sample_weight': 0.3
            },
            'middle_income': {
                'zones': ['Tatuapé', 'Santana', 'Lapa'],
                'expected_density': 'medium',
                'sample_weight': 0.4
            },
            'low_income': {
                'zones': ['Zona Leste', 'Zona Sul'],
                'expected_density': 'low',
                'sample_weight': 0.3
            }
        }
    
    def create_grid(self, cell_size_km: float = 1.0) -> gpd.GeoDataFrame:
        """
        Cria grid de células sobre São Paulo
        
        Args:
            cell_size_km: Tamanho da célula em km
        
        Returns:
            GeoDataFrame com as células do grid
        """
        # Conversão aproximada: 1° lat ≈ 111 km, 1° lon ≈ 96 km (em SP)
        lat_step = cell_size_km / 111
        lon_step = cell_size_km / 96
        
        lats = np.arange(self.sp_bounds['lat_min'], 
                        self.sp_bounds['lat_max'], 
                        lat_step)
        lons = np.arange(self.sp_bounds['lon_min'], 
                        self.sp_bounds['lon_max'], 
                        lon_step)
        
        cells = []
        cell_id = 0
        
        for i, lat in enumerate(lats[:-1]):
            for j, lon in enumerate(lons[:-1]):
                # Criar polígono da célula
                cell_polygon = box(lon, lat, 
                                  lons[j+1], lats[i+1])
                
                cells.append({
                    'cell_id': cell_id,
                    'geometry': cell_polygon,
                    'center_lat': (lat + lats[i+1]) / 2,
                    'center_lon': (lon + lons[j+1]) / 2,
                    'area_km2': cell_size_km ** 2
                })
                cell_id += 1
        
        gdf = gpd.GeoDataFrame(cells, crs='EPSG:4326')
        return gdf
    
    def stratified_sample(self, 
                         grid: gpd.GeoDataFrame, 
                         total_samples: int = 100,
                         random_seed: int = 42) -> gpd.GeoDataFrame:
        """
        Realiza amostragem estratificada
        
        Args:
            grid: GeoDataFrame com o grid
            total_samples: Número total de amostras desejadas
            random_seed: Seed para reprodutibilidade
        
        Returns:
            GeoDataFrame com células amostradas
        """
        np.random.seed(random_seed)
        
        # Atribuir estratos baseado em localização (simplificado)
        # Em produção, usar dados reais de renda/densidade
        grid['stratum'] = 'middle_income'
        
        # Zona oeste (mais rica)
        grid.loc[grid['center_lon'] < -46.65, 'stratum'] = 'high_income'
        
        # Zona leste (mais periférica)
        grid.loc[grid['center_lon'] > -46.55, 'stratum'] = 'low_income'
        
        # Calcular número de amostras por estrato
        samples_per_stratum = {}
        for stratum_name, stratum_info in self.strata.items():
            n_samples = int(total_samples * stratum_info['sample_weight'])
            samples_per_stratum[stratum_name] = n_samples
        
        # Amostrar de cada estrato
        sampled_cells = []
        for stratum_name, n_samples in samples_per_stratum.items():
            stratum_cells = grid[grid['stratum'] == stratum_name]
            
            if len(stratum_cells) > 0:
                sample_size = min(n_samples, len(stratum_cells))
                sampled = stratum_cells.sample(n=sample_size, random_state=random_seed)
                sampled_cells.append(sampled)
        
        return pd.concat(sampled_cells, ignore_index=True)
    
    def visualize_sampling(self, sampled_grid: gpd.GeoDataFrame, 
                          output_path: str = 'sampling_map.html'):
        """
        Cria mapa interativo da amostragem
        """
        # Centro de São Paulo
        m = folium.Map(location=[-23.55, -46.63], zoom_start=11)
        
        # Cores por estrato
        colors = {
            'high_income': 'green',
            'middle_income': 'blue',
            'low_income': 'red'
        }
        
        # Adicionar células amostradas
        for idx, row in sampled_grid.iterrows():
            folium.Rectangle(
                bounds=[[row.geometry.bounds[1], row.geometry.bounds[0]],
                       [row.geometry.bounds[3], row.geometry.bounds[2]]],
                color=colors.get(row['stratum'], 'gray'),
                fill=True,
                fillOpacity=0.3,
                popup=f"Cell {row['cell_id']}<br>Stratum: {row['stratum']}"
            ).add_to(m)
        
        m.save(output_path)
        print(f"Mapa salvo em: {output_path}")


class SatelliteImageDownloader:
    """
    Download de imagens de satélite via diferentes APIs
    """
    
    def __init__(self, api_key: str = None, provider: str = 'google'):
        """
        Args:
            api_key: Chave da API (Google Maps, Bing, etc)
            provider: 'google', 'bing', ou 'esri'
        """
        self.api_key = api_key
        self.provider = provider
        self.output_dir = Path('satellite_images')
        self.output_dir.mkdir(exist_ok=True)
    
    def download_tile_google(self, 
                            lat: float, 
                            lon: float, 
                            zoom: int = 20,
                            size: str = "640x640") -> str:
        """
        Download via Google Maps Static API
        
        Args:
            lat, lon: Coordenadas do centro
            zoom: Nível de zoom (20 = máxima resolução ~30cm/pixel)
            size: Tamanho da imagem
        
        Returns:
            Caminho do arquivo salvo
        """
        if not self.api_key:
            raise ValueError("API key necessária para Google Maps")
        
        url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            'center': f"{lat},{lon}",
            'zoom': zoom,
            'size': size,
            'maptype': 'satellite',
            'key': self.api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            filename = self.output_dir / f"tile_{lat}_{lon}_z{zoom}.png"
            with open(filename, 'wb') as f:
                f.write(response.content)
            return str(filename)
        else:
            raise Exception(f"Erro no download: {response.status_code}")
    
    def download_tile_esri(self,
                          lat: float,
                          lon: float,
                          width: int = 640,
                          height: int = 640) -> str:
        """
        Download via ESRI World Imagery (gratuito, mas menor resolução)
        """
        # Converter para Web Mercator
        import math
        
        def latlon_to_meters(lat, lon):
            x = lon * 20037508.34 / 180
            y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
            y = y * 20037508.34 / 180
            return x, y
        
        x, y = latlon_to_meters(lat, lon)
        
        # Calcular bbox (aproximado para 500m x 500m)
        buffer = 250  # metros
        bbox = f"{x-buffer},{y-buffer},{x+buffer},{y+buffer}"
        
        url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
        params = {
            'bbox': bbox,
            'bboxSR': '3857',
            'size': f"{width},{height}",
            'imageSR': '3857',
            'format': 'png',
            'f': 'image'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            filename = self.output_dir / f"esri_tile_{lat}_{lon}.png"
            with open(filename, 'wb') as f:
                f.write(response.content)
            return str(filename)
        else:
            raise Exception(f"Erro no download ESRI: {response.status_code}")
    
    def download_batch(self, 
                      sampled_grid: gpd.GeoDataFrame,
                      delay: float = 0.5) -> pd.DataFrame:
        """
        Download em batch com controle de rate limit
        
        Args:
            sampled_grid: GeoDataFrame com células amostradas
            delay: Delay entre requests (segundos)
        
        Returns:
            DataFrame com metadados dos downloads
        """
        results = []
        
        for idx, row in sampled_grid.iterrows():
            try:
                print(f"Downloading tile {idx+1}/{len(sampled_grid)}...")
                
                if self.provider == 'google' and self.api_key:
                    filepath = self.download_tile_google(
                        row['center_lat'], 
                        row['center_lon']
                    )
                elif self.provider == 'esri':
                    filepath = self.download_tile_esri(
                        row['center_lat'],
                        row['center_lon']
                    )
                else:
                    print(f"Provider {self.provider} sem API key, pulando...")
                    continue
                
                results.append({
                    'cell_id': row['cell_id'],
                    'lat': row['center_lat'],
                    'lon': row['center_lon'],
                    'stratum': row['stratum'],
                    'filepath': filepath,
                    'status': 'success'
                })
                
                time.sleep(delay)
                
            except Exception as e:
                print(f"Erro na célula {row['cell_id']}: {e}")
                results.append({
                    'cell_id': row['cell_id'],
                    'lat': row['center_lat'],
                    'lon': row['center_lon'],
                    'stratum': row['stratum'],
                    'filepath': None,
                    'status': f'error: {str(e)}'
                })
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(self.output_dir / 'download_metadata.csv', index=False)
        
        return df_results


# ============== EXEMPLO DE USO ==============

if __name__ == "__main__":
    # 1. Criar sampler
    sampler = SaoPauloSampler()
    
    # 2. Criar grid de 1km²
    print("Criando grid de São Paulo...")
    grid = sampler.create_grid(cell_size_km=0.5)
    print(f"Grid criado com {len(grid)} células")
    
    # 3. Amostragem estratificada
    print("\nRealizando amostragem estratificada...")
    sampled = sampler.stratified_sample(grid, total_samples=600, random_seed=42)
    print(f"Amostradas {len(sampled)} células")
    print(f"\nDistribuição por estrato:")
    print(sampled['stratum'].value_counts())
    
    # 4. Visualizar amostragem
    print("\nCriando mapa de visualização...")
    sampler.visualize_sampling(sampled)
    
    # 5. Exportar coordenadas para uso posterior
    sampled[['cell_id', 'center_lat', 'center_lon', 'stratum']].to_csv(
        'sampled_coordinates.csv', 
        index=False
    )
    print("\nCoordenadas exportadas para 'sampled_coordinates.csv'")
