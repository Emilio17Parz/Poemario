import os
import json
import shutil

# 1. Definimos la ruta de la carpeta donde están los archivos
ruta_directorio = r"F:\Jair\Semestre7\6CV1-Machine Learning\DatasetPoemas\datasets\Terceto"

# 2. Definimos la ruta de destino ("cortar")
ruta_destino = os.path.join(ruta_directorio, "cortar")

# Creamos la carpeta "cortar" si no existe
if not os.path.exists(ruta_destino):
    os.makedirs(ruta_destino)
    print(f"Carpeta creada: {ruta_destino}")

# Contadores para el reporte final
archivos_movidos = 0
archivos_correctos = 0
errores = 0

print("--- Iniciando análisis de archivos ---\n")

# 3. Recorremos todos los archivos en el directorio
for archivo in os.listdir(ruta_directorio):
    ruta_completa = os.path.join(ruta_directorio, archivo)

    # Verificamos que sea un archivo y que termine en .json (para ignorar la carpeta 'cortar')
    if os.path.isfile(ruta_completa) and archivo.lower().endswith('.json'):
        try:
            # Abrimos y leemos el JSON
            with open(ruta_completa, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 4. Extraemos el tipo de poema. Usamos .get para evitar errores si falta la llave
            # Estructura: data -> "poema" -> "tipo"
            poema_data = data.get("poema", {})
            tipo_poema = poema_data.get("tipo", "Desconocido")

            # 5. Condición: Si NO es Redondilla, lo movemos
            if tipo_poema != "Terceto":
                shutil.move(ruta_completa, os.path.join(ruta_destino, archivo))
                print(f"[MOVIDO] {archivo} -> Era tipo: '{tipo_poema}'")
                archivos_movidos += 1
            else:
                # Si es Redondilla, se queda ahí
                archivos_correctos += 1

        except json.JSONDecodeError:
            print(f"[ERROR JSON] No se pudo leer {archivo}. Formato inválido.")
            errores += 1
        except Exception as e:
            print(f"[ERROR] Problema con {archivo}: {e}")
            errores += 1

print("\n--- Resumen ---")
print(f"Archivos que se quedaron (Redondillas correctas): {archivos_correctos}")
print(f"Archivos movidos a 'cortar' (Incorrectos): {archivos_movidos}")
print(f"Archivos con errores de lectura: {errores}")