import os
import json
import hashlib
import re
from jsonschema import validate, ValidationError

# --- CONFIGURACIÓN ---
BASE_DIR = r"C:\Users\jecal\Poemario"
DATASET_PATH = os.path.join(BASE_DIR, "datasets")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema", "poema.schema.json")

# --- UTILIDADES ---

def load_schema():
    """Carga el esquema JSON para validación."""
    if not os.path.exists(SCHEMA_PATH):
        print(f"[ERROR CRITICO] No se encuentra el esquema en {SCHEMA_PATH}")
        exit(1)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_content_fingerprint(text):
    """
    Genera un hash único del contenido del poema.
    Normaliza el texto (minúsculas y espacios simples) para detectar
    duplicados reales aunque tengan formato ligeramente distinto.
    """
    if not text: return None
    # 1. Minúsculas
    text = text.lower()
    # 2. Colapsar espacios y saltos de línea ( "hola   mundo" == "hola mundo")
    text = re.sub(r'\s+', ' ', text).strip()
    # 3. Hash SHA256
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def scan_validate_clean():
    print(f"Iniciando Validacion y Limpieza en: {DATASET_PATH}")
    print("-" * 50)
    
    schema = load_schema()
    hashes_vistos = {} # hash -> ruta_archivo
    
    stats = {
        "procesados": 0,
        "validos": 0,
        "duplicados_borrados": 0,
        "errores_schema": 0,
        "errores_lectura": 0
    }

    # Recorrer recursivamente
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if not file.endswith(".json"): continue
            
            filepath = os.path.join(root, file)
            stats["procesados"] += 1

            try:
                # 1. Cargar JSON
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 2. Validar contra el Schema
                validate(instance=data, schema=schema)

                # 3. Extraer texto del poema para chequear duplicados
                # Asumimos que el schema garantiza que data['poema']['texto'] existe
                texto_poema = data.get("poema", {}).get("texto", "")
                
                fingerprint = get_content_fingerprint(texto_poema)

                if not fingerprint:
                    print(f"[ADVERTENCIA] Poema sin texto en {file}. Borrando...")
                    os.remove(filepath)
                    stats["errores_schema"] += 1
                    continue

                # 4. Verificar si ya existe este contenido
                if fingerprint in hashes_vistos:
                    original = hashes_vistos[fingerprint]
                    # Es un duplicado. Lo borramos.
                    print(f"[DUPLICADO] Borrando {file}")
                    print(f"   -> Identico a: {os.path.basename(original)}")
                    os.remove(filepath)
                    stats["duplicados_borrados"] += 1
                else:
                    # Es único y válido. Lo registramos.
                    hashes_vistos[fingerprint] = filepath
                    stats["validos"] += 1

            except json.JSONDecodeError:
                print(f"[JSON INVALIDO] Borrando archivo corrupto: {file}")
                os.remove(filepath)
                stats["errores_lectura"] += 1
            except ValidationError as e:
                print(f"[ERROR SCHEMA] {file} no cumple las reglas.")
                print(f"   -> Detalle: {e.message}")
                # Opcional: Borrar si no cumple schema
                # os.remove(filepath) 
                stats["errores_schema"] += 1
            except Exception as ex:
                print(f"[ERROR] Error inesperado en {file}: {ex}")
                stats["errores_lectura"] += 1

    print("\n" + "="*40)
    print("RESUMEN FINAL")
    print("="*40)
    print(f"Total procesados:      {stats['procesados']}")
    print(f"Validos y unicos:      {stats['validos']}")
    print(f"Duplicados borrados:   {stats['duplicados_borrados']}")
    print(f"Errores de Schema:     {stats['errores_schema']}")
    print(f"Errores de Lectura:    {stats['errores_lectura']}")
    print("="*40)

if __name__ == "__main__":
    scan_validate_clean()