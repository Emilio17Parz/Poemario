import json
import os
import glob

# Ruta donde están los archivos actualmente
folder = os.path.join("datasets", "Haiku")

# Buscar los archivos JSON en esa carpeta
files = glob.glob(os.path.join(folder, "*.json"))

print(f"Analizando archivos en: {folder}")

for file_path in files:
    filename = os.path.basename(file_path)
    
    # Leemos el archivo
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Error leyendo {filename}")
            continue

    # Si el contenido es una LISTA (que es el error actual), lo separamos
    if isinstance(content, list):
        print(f"⚠️ Separando archivo gigante: {filename} ({len(content)} poemas)...")
        
        # Obtenemos el nombre base (ej. "Asombro")
        base_name = os.path.splitext(filename)[0]
        
        # Creamos un archivo por cada poema en la lista
        for i, poem in enumerate(content):
            # Nombre nuevo: Asombro_001.json, Asombro_002.json, etc.
            new_filename = f"{base_name}_{i+1:03d}.json"
            new_path = os.path.join(folder, new_filename)
            
            with open(new_path, 'w', encoding='utf-8') as f_out:
                json.dump(poem, f_out, indent=4, ensure_ascii=False)
        
        # IMPORTANTE: Cerramos el archivo y lo borramos porque el grande da error
        f.close()
        os.remove(file_path) 
        print(f"✅ Se generaron {len(content)} archivos y se eliminó el gigante {filename}.")

    else:
        print(f"ℹ️ El archivo {filename} ya es un objeto único (correcto).")

print("\n¡Reparación terminada! Ahora corre el validador.")