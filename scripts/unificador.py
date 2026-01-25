import os
import json
import re
import unicodedata
import hashlib
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
# --- CONFIGURACIÓN DE RUTAS CORREGIDA ---
BASE_PATH = Path(r"C:\Users\calza\Poemario")
BASE_PATH_V2 = Path(r"C:\Users\calza\Poemariov2")

INPUT_FOLDERS = [
    BASE_PATH / "datasets2",                   # Carpeta en Poemario
    BASE_PATH_V2 / "datasets",                # Carpeta en Poemariov2
    BASE_PATH_V2 / "dataset_decima",          # Carpeta en Poemariov2
    BASE_PATH_V2 / "dataset_triolet"          # Carpeta en Poemariov2
]

OUTPUT_STRUCT = BASE_PATH / 'dataset3_estructural'
OUTPUT_FULL = BASE_PATH / 'dataset3_completo'

# --- MAPEADO DE LAS 49 CATEGORÍAS ---

# 1. Formas con métrica de líneas estricta o casi estricta
FORMAS_ESTRICTAS = {
    "soneto": 14, "haiku": 3, "tanka": 5, "decima_espinela": 10, "decima": 10,
    "triolet": 8, "villanelle": 19, "limerick": 5, "pareado": 2, "terceto": 3,
    "cuarteto": 4, "cuarteta": 4, "redondilla": 4, "serventesio": 4, "copla": 4,
    "seguidilla": 4, "estrofa_safica": 4, "estrofa_alcaica": 4, "rondeau": 13,
    "rondo": 13, "sestina": 39, "pantoum": 16
}

# 2. Formas de contenido, métrica libre o estructuras complejas (sin conteo fijo único)
CATEGORIAS_LIBRES = [
    "acrostico", "balada", "cancion_petrarquista", "egloga", "elegia", "epigrama",
    "estancia", "gacela", "ghazal", "himno", "jarcha", "moaxaja", "oda", 
    "palindromo_poetico", "poema_concreto", "poema_didactico", "poema_dramatico",
    "poema_elegiaco", "poema_en_prosa", "poema_epico", "poema_generico",
    "poema_lirico", "poema_narrativo", "poema_religioso", "poema_satirico",
    "romance", "silva", "terceto_encadenado", "versiculo", "verso_libre", "zejel"
]

# --- UTILIDADES ---

def normalize_text_for_hash(text):
    if not text: return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()
    return re.sub(r'[^a-z0-9]', '', text)

def get_content_hash(text):
    return hashlib.md5(normalize_text_for_hash(text).encode('utf-8')).hexdigest()

def normalize_cat(text):
    if not text: return "desconocido"
    t = text.lower().strip()
    t = unicodedata.normalize('NFKD', t).encode('ASCII', 'ignore').decode('utf-8')
    return re.sub(r'[^a-z0-9]', '_', t).strip('_')

def validate_structure(category, lines_count):
    cat = normalize_cat(category)
    if cat in FORMAS_ESTRICTAS:
        # Si es un terceto encadenado, la lógica suele ser 3n + 1 (ej: 10, 13, 16...)
        if cat == "terceto_encadenado":
            return (lines_count >= 4 and lines_count % 3 == 1, "terceto_encadenado_irregular")
        return (lines_count == FORMAS_ESTRICTAS[cat], f"{cat}_irregular")
    return (lines_count > 0, "archivo_vacio")

# --- PROCESO ---

def start_process():
    print(f"--- Iniciando Procesamiento de 49 Clases ---")
    seen_hashes = set()
    stats = {"procesados": 0, "errores": 0, "duplicados": 0}

    for folder in INPUT_FOLDERS:
        if not folder.exists(): continue
        print(f"📂 Escaneando {folder.name}...")

        for root, _, files in os.walk(folder):
            folder_hint = normalize_cat(os.path.basename(root))
            for file in files:
                if not file.endswith('.json'): continue
                
                try:
                    path = Path(root) / file
                    with open(path, 'r', encoding='utf-8') as f:
                        raw = f.read()
                        try:
                            data = json.loads(raw)
                        except:
                            data = json.loads(raw.replace("'", '"'))

                    poema = data.get('poema', {})
                    texto = poema.get('texto', '')
                    if not texto: continue

                    # Hash para evitar duplicados
                    h = get_content_hash(texto)
                    if h in seen_hashes:
                        stats["duplicados"] += 1
                        continue

                    # Clasificación
                    tipo_original = normalize_cat(poema.get('tipo', 'desconocido'))
                    # Priorizar carpeta si el tipo es genérico
                    if tipo_original in ['desconocido', 'poema_generico'] and folder_hint not in ['datasets2', 'datasets']:
                        tipo_original = folder_hint

                    lines_count = len([l for l in texto.split('\n') if l.strip()])
                    is_valid, fallback = validate_structure(tipo_original, lines_count)

                    # Respeto al formato: No borramos nada, solo agregamos metadatos de control
                    data['unificador_meta'] = {"hash": h, "lineas": lines_count, "valido": is_valid}
                    
                    # 1. Guardado en Dataset Completo (Mantiene categorías originales)
                    cat_full = tipo_original if is_valid or tipo_original in CATEGORIAS_LIBRES else f"revisar_{tipo_original}"
                    path_full = OUTPUT_FULL / cat_full
                    path_full.mkdir(parents=True, exist_ok=True)
                    
                    # 2. Guardado en Dataset Estructural (Métrica rigurosa)
                    cat_struct = tipo_original
                    if tipo_original in CATEGORIAS_LIBRES:
                        cat_struct = "verso_libre_largo" if lines_count > 14 else "verso_libre_corto"
                    elif not is_valid:
                        cat_struct = fallback
                    
                    path_struct = OUTPUT_STRUCT / cat_struct
                    path_struct.mkdir(parents=True, exist_ok=True)

                    # Escribir archivos (nombre único para evitar sobrescribir poemas con mismo título)
                    filename = f"{h[:10]}_{file}"
                    for p in [path_full, path_struct]:
                        with open(p / filename, 'w', encoding='utf-8') as out:
                            json.dump(data, out, ensure_ascii=False, indent=2)

                    seen_hashes.add(h)
                    stats["procesados"] += 1
                    if stats["procesados"] % 500 == 0:
                        print(f"📊 Avance: {stats['procesados']} poemas únicos guardados...")

                except:
                    stats["errores"] += 1

    print(f"\n✅ PROCESO COMPLETADO\n- Guardados: {stats['procesados']}\n- Duplicados saltados: {stats['duplicados']}\n- Errores: {stats['errores']}")

if __name__ == "__main__":
    start_process()