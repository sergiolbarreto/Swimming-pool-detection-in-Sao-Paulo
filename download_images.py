import pandas as pd
import requests
import time
from pathlib import Path
import argparse
from PIL import Image
import io
import math


class ImageDownloader:
    def __init__(self, provider='esri', api_key=None):
        self.provider = provider
        self.api_key = api_key
        self.output_dir = Path('dataset/raw_images')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_google(self, lat, lon, cell_id, zoom=20, size="640x640"):
        """Download via Google Maps Static API"""
        if not self.api_key:
            raise ValueError("Google Maps requer API key!")
        
        url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            'center': f"{lat},{lon}",
            'zoom': zoom,
            'size': size,
            'maptype': 'satellite',
            'key': self.api_key,
            'scale': 2  # Alta resolução
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                filename = self.output_dir / f"cell_{cell_id:04d}.jpg"
                
                # Converter para JPG e salvar
                img = Image.open(io.BytesIO(response.content))
                img = img.convert('RGB')
                img.save(filename, 'JPEG', quality=95)
                
                return str(filename), True
            else:
                error_msg = response.text[:100] if response.text else 'Unknown error'
                print(f"❌ Erro Google: {response.status_code} - {error_msg}")
                return None, False
        except Exception as e:
            print(f"❌ Erro ao baixar: {e}")
            return None, False
    
    def download_esri(self, lat, lon, cell_id, width=640, height=640):
        """Download via ESRI World Imagery (gratuito)"""
        
        # Converter lat/lon para Web Mercator
        x = lon * 20037508.34 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        y = y * 20037508.34 / 180
        
        # Bbox de ~500m x 500m
        buffer = 300
        bbox = f"{x-buffer},{y-buffer},{x+buffer},{y+buffer}"
        
        url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
        params = {
            'bbox': bbox,
            'bboxSR': '3857',
            'size': f"{width},{height}",
            'imageSR': '3857',
            'format': 'jpg',
            'f': 'image'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                filename = self.output_dir / f"cell_{cell_id:04d}.jpg"
                
                img = Image.open(io.BytesIO(response.content))
                img.save(filename, 'JPEG', quality=95)
                
                return str(filename), True
            else:
                print(f"❌ Erro ESRI: {response.status_code}")
                return None, False
                
        except Exception as e:
            print(f"❌ Erro ao baixar célula {cell_id}: {e}")
            return None, False
    
    def download_batch(self, csv_path='sampled_coordinates.csv', 
                      max_samples=None, delay=1.0):
        """
        Download em batch das coordenadas
        
        Args:
            csv_path: Caminho do CSV com coordenadas
            max_samples: Número máximo de amostras (None = todas)
            delay: Delay entre requests (segundos)
        """
        # Carregar coordenadas
        df = pd.read_csv(csv_path)
        
        if max_samples:
            df = df.head(max_samples)
        
        print(f"\n{'='*60}")
        print(f"📥 DOWNLOAD DE IMAGENS - Provider: {self.provider.upper()}")
        print(f"{'='*60}")
        print(f"Total de amostras: {len(df)}")
        print(f"Destino: {self.output_dir}/")
        print(f"{'='*60}\n")
        
        results = []
        success_count = 0
        
        for idx, row in df.iterrows():
            cell_id = row['cell_id']
            lat = row['center_lat']
            lon = row['center_lon']
            stratum = row['stratum']
            
            print(f"[{idx+1}/{len(df)}] Baixando célula {cell_id} ({stratum})...", end=' ')
            
            # Download baseado no provider
            if self.provider == 'google':
                filepath, success = self.download_google(lat, lon, cell_id)
            elif self.provider == 'esri':
                filepath, success = self.download_esri(lat, lon, cell_id)
            else:
                print(f"❌ Provider '{self.provider}' desconhecido")
                continue
            
            if success:
                print("✅")
                success_count += 1
            else:
                print("❌")
            
            results.append({
                'cell_id': cell_id,
                'lat': lat,
                'lon': lon,
                'stratum': stratum,
                'filepath': filepath,
                'success': success
            })
            
            # Rate limiting
            time.sleep(delay)
        
        # Salvar metadata
        df_results = pd.DataFrame(results)
        metadata_path = self.output_dir.parent / 'download_metadata.csv'
        df_results.to_csv(metadata_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"✅ Download concluído!")
        print(f"{'='*60}")
        print(f"Sucesso: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
        print(f"Imagens salvas em: {self.output_dir}/")
        print(f"Metadata salvo em: {metadata_path}")
        print(f"{'='*60}\n")
        
        return df_results


def main():
    parser = argparse.ArgumentParser(
        description='Download de imagens de satélite para detecção de piscinas'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        default='esri',
        choices=['esri', 'google'],
        help='Provedor de imagens (esri=grátis, google=pago mas melhor)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key do Google Maps (necessário para provider=google)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Número máximo de amostras a baixar (None = todas)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay entre requests em segundos (default: 1.0)'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default='sampled_coordinates.csv',
        help='Caminho do CSV com coordenadas'
    )
    
    args = parser.parse_args()
    
    # Validar args
    if args.provider == 'google' and not args.api_key:
        print("❌ ERRO: Provider 'google' requer --api-key\n")
        print("Como obter API key:")
        print("1. Acesse: https://console.cloud.google.com")
        print("2. Crie um projeto")
        print("3. Ative 'Maps Static API'")
        print("4. Crie uma credencial (API key)")
        print("5. Use: python download_images.py --provider google --api-key SUA_KEY\n")
        return
    
    # Download
    downloader = ImageDownloader(
        provider=args.provider,
        api_key=args.api_key
    )
    
    results = downloader.download_batch(
        csv_path=args.csv,
        max_samples=args.samples,
        delay=args.delay
    )
    
    # Estatísticas por estrato
    if len(results) > 0:
        print("\n📊 Estatísticas por estrato:")
        print(results.groupby('stratum')['success'].agg(['sum', 'count']))


if __name__ == "__main__":
    main()