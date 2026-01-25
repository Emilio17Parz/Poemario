import os
import json
import hashlib
import re
import unicodedata
import shutil
from jsonschema import validate, ValidationError

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = r"C:\Users\calza\Poemariov2" # Tu directorio base actual

# Definimos todas las carpetas que queremos escanear como orígenes
SOURCES = [
    {"path": r"C:\Users\calza\Respaldos\Poemariov2\Poemario\datasets", "label": "BACKUP_EXTERNO"},
    {"path": os.path.join(BASE_DIR, "datasets"), "label": "LOCAL_MAIN"},
    {"path": os.path.join(BASE_DIR, "dataset_decima"), "label": "DECIMA"},
    {"path": os.path.join(BASE_DIR, "dataset_triolet"), "label": "TRIOLET"},
    # Puedes agregar más rutas aquí fácilmente
]

# RUTA DE SALIDA
DEST_FINAL = os.path.join(BASE_DIR, "datasets_unificados")

# SCHEMA
SCHEMA_PATH = os.path.join(BASE_DIR, "schema", "poema.schema.json")

# --- UTILIDADES ---

def load_schema():
    if not os.path.exists(SCHEMA_PATH):
        print(f"[ERROR] No se encuentra el esquema en {SCHEMA_PATH}")
        return None
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_text(text):
    if not text: return ""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_content_fingerprint(text):
    norm_text = normalize_text(text)
    if not norm_text: return None
    return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

def calculate_quality_score(data):
    score = 0
    poema = data.get("poema", {})
    if poema.get("titulo"): score += 10
    if poema.get("autor"): score += 10
    if poema.get("fecha"): score += 5
    if data.get("meta", {}).get("etiquetas"): score += len(data["meta"]["etiquetas"])
    if data.get("meta", {}).get("fuente"): score += 2
    if len(poema.get("texto", "")) > 50: score += 1
    return score

# --- LÓGICA DE UNIFICACIÓN ---

def ingest_folder(root_base_path, registry, schema, source_label):
    if not os.path.exists(root_base_path):
        print(f"[{source_label}] Carpeta no encontrada: {root_base_path} (Saltando...)")
        return

    print(f" -> Escaneando {source_label}: {root_base_path}")
    count = 0
    
    for current_root, dirs, files in os.walk(root_base_path):
        for file in files:
            if not file.endswith(".json"): continue
            
            full_path = os.path.join(current_root, file)
            rel_path = os.path.relpath(full_path, root_base_path)

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if schema:
                    validate(instance=data, schema=schema)

                text = data.get("poema", {}).get("texto", "")
                fingerprint = get_content_fingerprint(text)
                
                if not fingerprint: continue 

                score = calculate_quality_score(data)

                # Regla de oro: Si es nuevo o tiene mejor calidad, se queda
                if fingerprint not in registry or score > registry[fingerprint]['score']:
                    registry[fingerprint] = {
                        'source_path': full_path,
                        'rel_path': rel_path,
                        'score': score,
                        'origin': source_label
                    }
                count += 1
            except (ValidationError, json.JSONDecodeError):
                pass 
            except Exception as e:
                print(f"   [WARN] Error en {file}: {e}")
    
    print(f"    - Archivos procesados en {source_label}: {count}")

def save_merged_dataset_with_structure(registry):
    if not os.path.exists(DEST_FINAL):
        os.makedirs(DEST_FINAL)
        print(f"\n[INFO] Carpeta de salida creada: {DEST_FINAL}")

    print(" -> Copiando archivos únicos a la carpeta de destino...")
    saved_count = 0

    for fingerprint, info in registry.items():
        src = info['source_path']
        rel_path = info['rel_path']
        
        dest_full_path = os.path.join(DEST_FINAL, rel_path)
        
        # Asegurar subcarpetas
        dest_folder = os.path.dirname(dest_full_path)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        
        # Evitar sobreescritura si el nombre de archivo coincide pero el contenido no
        base_dest, ext = os.path.splitext(dest_full_path)
        counter = 1
        while os.path.exists(dest_full_path):
            dest_full_path = f"{base_dest}_{counter}{ext}"
            counter += 1
        
        shutil.copy2(src, dest_full_path)
        saved_count += 1

    return saved_count

def main():
    print("="*60)
    print("SISTEMA DE UNIFICACIÓN DE POEMARIOS V2")
    print("="*60)
    
    schema = load_schema()
    if not schema: return

    master_registry = {}

    # Procesar cada fuente configurada
    for source in SOURCES:
        ingest_folder(source["path"], master_registry, schema, source["label"])

    print("-" * 60)
    print(f"Total de poemas únicos (por huella digital): {len(master_registry)}")
    
    total_saved = save_merged_dataset_with_structure(master_registry)

    print("="*60)
    print(f"PROCESO TERMINADO")
    print(f"Destino: {DEST_FINAL}")
    print(f"Archivos finales guardados: {total_saved}")
    print("="*60)

if __name__ == "__main__":
    main()