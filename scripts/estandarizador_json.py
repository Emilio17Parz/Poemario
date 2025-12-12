import os
import json

# --- CONFIGURACION ---
BASE_DIR = r"C:\Users\jecal\Poemario"
DATASET_PATH = os.path.join(BASE_DIR, "datasets")
SCHEMA_REF = "http://json-schema.org/draft-07/schema#"
LOG_INTERVAL = 100  # Cada cuantos archivos imprime el progreso en pantalla

def limpiar_estructura():
    print(f"Iniciando estandarizacion en: {DATASET_PATH}")
    print("-" * 60)

    stats = {
        "procesados": 0,
        "corregidos": 0,
        "eliminados_por_incompletos": 0,
        "errores": 0
    }

    # Variable para controlar el log de cambio de carpeta
    carpeta_actual_log = ""

    for root, dirs, files in os.walk(DATASET_PATH):
        # Log visual cuando cambia de carpeta
        nombre_carpeta = os.path.basename(root)
        if nombre_carpeta != carpeta_actual_log and nombre_carpeta != "datasets":
            print(f"[INFO] Entrando a carpeta: {nombre_carpeta}")
            carpeta_actual_log = nombre_carpeta

        for file in files:
            if not file.endswith(".json"): continue
            
            filepath = os.path.join(root, file)
            stats["procesados"] += 1
            
            # Log de progreso periodico
            if stats["procesados"] % LOG_INTERVAL == 0:
                print(f"   ... Procesando archivo #{stats['procesados']} ({file})")

            # Paso 1: Lectura de datos (Abrir y cerrar inmediatamente para liberar el archivo)
            data = None
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[ERROR LECTURA] No se pudo leer {file}: {e}")
                stats["errores"] += 1
                continue # Saltamos al siguiente archivo si no se puede leer

            # Paso 2: Validacion y Logica
            try:
                subcategoria = data.get("subcategoria")
                poema_obj = data.get("poema", {})
                texto = poema_obj.get("texto")
                tipo = poema_obj.get("tipo")
                keyword = poema_obj.get("palabra_clave_ingresada")

                # Validacion de campos obligatorios
                if not all([subcategoria, texto, tipo]):
                    print(f"[ELIMINADO] Archivo incompleto: {file}")
                    # Como el archivo ya esta cerrado (fuera del 'with'), podemos borrarlo sin error
                    try:
                        os.remove(filepath)
                        stats["eliminados_por_incompletos"] += 1
                    except OSError as e:
                        print(f"[ERROR SISTEMA] No se pudo borrar {file}: {e}")
                    continue

                # Normalizacion de keyword
                if keyword is None:
                    keyword = ""

                # --- CONSTRUCCION DEL NUEVO JSON LIMPIO ---
                nuevo_json = {
                    "$schema": SCHEMA_REF,
                    "subcategoria": subcategoria,
                    "poema": {
                        "texto": texto,
                        "tipo": tipo,
                        "palabra_clave_ingresada": keyword
                    }
                }

                # Paso 3: Reescritura del archivo
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(nuevo_json, f, ensure_ascii=False, indent=2)
                
                stats["corregidos"] += 1

            except Exception as e:
                print(f"[ERROR PROCESO] Fallo al procesar {file}: {e}")
                stats["errores"] += 1

    print("\n" + "="*40)
    print("RESUMEN DE ESTANDARIZACION")
    print("="*40)
    print(f"Total archivos escaneados:       {stats['procesados']}")
    print(f"Archivos limpiados/reescritos:   {stats['corregidos']}")
    print(f"Archivos eliminados (incompletos): {stats['eliminados_por_incompletos']}")
    print(f"Errores generales:               {stats['errores']}")
    print("="*40)

if __name__ == "__main__":
    limpiar_estructura()