import os
from PIL import Image

dataset_path = r"C:\Users\calza\Poemario\datasetimg"
basura_path = r"C:\Users\calza\Poemario\archivos_corruptos"
os.makedirs(basura_path, exist_ok=True)

print("🚀 Limpiando dataset...")

eliminados = 0
for root, _, files in os.walk(dataset_path):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, f)
            try:
                with Image.open(path) as img:
                    img.verify() # Verifica que el archivo no esté truncado
            except (IOError, SyntaxError) as e:
                print(f"❌ Corrupto detectado: {f}")
                # Lo movemos en lugar de borrarlo por seguridad
                os.rename(path, os.path.join(basura_path, f))
                eliminados += 1

print(f"--- Limpieza terminada ---")
print(f"✅ Archivos corruptos movidos a /archivos_corruptos: {eliminados}")