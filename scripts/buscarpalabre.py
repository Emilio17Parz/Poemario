import json
import os
from pathlib import Path

# --- CONFIGURACIÓN ---
# Cambia esto por la ruta donde tienes tus .json guardados
CARPETA_POEMAS = "./datasets" 

def generar_estructura_desde_archivos(ruta_carpeta):
    """
    Recorre todos los archivos JSON y crea una estructura de categorías única.
    """
    estructura = {}
    ruta = Path(ruta_carpeta)
    
    # Busca archivos .json recursivamente (en todas las subcarpetas)
    archivos_json = list(ruta.rglob("*.json"))
    
    if not archivos_json:
        print(f"❌ No se encontraron archivos JSON en {ruta_carpeta}")
        return {}

    print(f"📂 Analizando {len(archivos_json)} archivos...")

    for archivo in archivos_json:
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 1. Extraemos la Categoría (usando 'subcategoria' del schema)
                # Usamos .title() para que "cocina" y "Cocina" sean lo mismo
                categoria = data.get("subcategoria", "Sin Categoria").strip().title()
                
                # 2. Extraemos el Tema específico (usando 'palabra_clave_ingresada')
                # Accedemos al objeto anidado 'poema'
                poema_obj = data.get("poema", {})
                palabra_clave = poema_obj.get("palabra_clave_ingresada", "").strip().capitalize()
                
                # Solo procesamos si hay datos válidos
                if categoria and palabra_clave:
                    if categoria not in estructura:
                        estructura[categoria] = set() # Usamos set para evitar duplicados
                    
                    estructura[categoria].add(palabra_clave)
                    
        except Exception as e:
            print(f"⚠️ Error leyendo {archivo.name}: {e}")

    # Convertimos los sets a listas para que se pueda imprimir/usar
    estructura_final = {k: list(v) for k, v in estructura.items()}
    return estructura_final

def imprimir_resultado_formateado(estructura):
    """
    Imprime el diccionario con formato de código Python listo para copiar.
    """
    print("\n" + "="*40)
    print("✅ ESTRUCTURA GENERADA (Copia esto):")
    print("="*40 + "\n")
    
    print("estructura_dataset = {")
    
    for categoria, items in estructura.items():
        print(f'    "{categoria}": [')
        # Ordenamos alfabéticamente para que se vea ordenado
        for i, item in enumerate(sorted(items)):
            coma = "," if i < len(items) - 1 else ""
            print(f'        "{item}"{coma}')
        print("    ],")
        
    print("}")
    print("\n" + "="*40)

if __name__ == "__main__":
    # Asegúrate de que la carpeta exista antes de correr
    if os.path.exists(CARPETA_POEMAS):
        diccionario = generar_estructura_desde_archivos(CARPETA_POEMAS)
        imprimir_resultado_formateado(diccionario)
    else:
        print(f"❌ La carpeta '{CARPETA_POEMAS}' no existe. Edita la variable CARPETA_POEMAS.")