import json
import os
import hashlib
import glob

DATASET_PATH = "datasets"

# Campos permitidos según tu ejemplo correcto
ALLOWED_KEYS_ROOT = ["subcategoria", "poema"]
ALLOWED_KEYS_POEMA = ["texto", "tipo", "palabra_clave_ingresada"]

def clean_poema_structure(data):
    """Limpia un objeto poema para dejar solo lo necesario"""
    if "poema" not in data:
        return None
    
    # Crear estructura limpia
    clean_data = {
        "subcategoria": data.get("subcategoria", "Desconocido"),
        "poema": {
            "texto": data["poema"].get("texto", ""),
            "tipo": data["poema"].get("tipo", "Desconocido"),
            "palabra_clave_ingresada": data["poema"].get("palabra_clave_ingresada", "")
        }
    }
    return clean_data

def save_single_file(data, original_path, index=0):
    """Guarda un poema individual en un archivo nuevo"""
    # Generamos un nombre único basado en el contenido para evitar colisiones
    content_hash = hashlib.md5(data["poema"]["texto"].encode('utf-8')).hexdigest()[:8]
    folder = os.path.dirname(original_path)
    filename = os.path.basename(original_path).replace(".json", "")
    
    # Nuevo nombre: Original_hash.json
    new_filename = f"{filename}_{content_hash}.json"
    new_path = os.path.join(folder, new_filename)
    
    with open(new_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generado: {new_filename}")

def run_corrector():
    print("🛠️ Iniciando Auto-Corrección de Dataset...\n")
    
    files_modified = False
    
    # Recorremos recursivamente todos los JSON
    for ruta, _, archivos in os.walk(DATASET_PATH):
        for archivo in archivos:
            if archivo.endswith(".json"):
                full_path = os.path.join(ruta, archivo)
                
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # CASO 1: El archivo es una LISTA (El error que mostraste)
                    if isinstance(data, list):
                        print(f"⚠️ Detectada lista en {archivo}. Separando...")
                        for i, item in enumerate(data):
                            clean_item = clean_poema_structure(item)
                            if clean_item:
                                save_single_file(clean_item, full_path, i)
                        
                        # Borramos el archivo original que tenía la lista
                        os.remove(full_path)
                        print(f"🗑️ Archivo original eliminado: {archivo}")
                        files_modified = True

                    # CASO 2: Es un diccionario pero tiene campos sucios (como titulo)
                    elif isinstance(data, dict):
                        # Verificamos si tiene campos extra
                        current_keys = set(data.get("poema", {}).keys())
                        if "titulo" in current_keys:
                            print(f"🧹 Limpiando campos extra en {archivo}...")
                            clean_item = clean_poema_structure(data)
                            # Sobreescribimos el archivo
                            with open(full_path, "w", encoding="utf-8") as f:
                                json.dump(clean_item, f, indent=2, ensure_ascii=False)
                            files_modified = True

                except Exception as e:
                    print(f"❌ Error leyendo {archivo}: {e}")

    if files_modified:
        print("\n✨ Corrección finalizada. Archivos reestructurados.")
    else:
        print("\n✅ No se requirieron correcciones.")

if __name__ == "__main__":
    run_corrector()