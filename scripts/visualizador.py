import os
import glob
from collections import Counter
import matplotlib.pyplot as plt

# Ruta relativa desde donde ejecutas el script
ruta_base = 'datasets'

# Verificar que la carpeta exista
if os.path.exists(ruta_base):
    print(f"--> Buscando dentro de: {os.path.abspath(ruta_base)}")
    
    # Búsqueda recursiva de archivos JSON
    patron = os.path.join(ruta_base, '**', '*.json')
    archivos_json = glob.glob(patron, recursive=True)

    total_archivos = len(archivos_json)
    print(f"---------------------------------------")
    print(f"RESULTADO TOTAL: Se encontraron {total_archivos} archivos JSON.")
    print(f"---------------------------------------")

    if total_archivos > 0:
        # --- PASO 1: PROCESAMIENTO DE DATOS ---
        # Extraemos solo la ruta del directorio de cada archivo encontrado
        carpetas = [os.path.dirname(archivo) for archivo in archivos_json]
        
        # Usamos Counter para contar cuántas veces se repite cada carpeta
        conteo_por_carpeta = Counter(carpetas)

        # Imprimimos el desglose en texto
        print("\nDesglose por carpeta:")
        for carpeta, cantidad in conteo_por_carpeta.items():
            # Mostramos la ruta relativa para que sea más legible
            nombre_corto = os.path.relpath(carpeta, start=os.getcwd())
            print(f" - {nombre_corto}: {cantidad} archivos")

        # --- PASO 2: GENERACIÓN DE LA GRÁFICA ---
        # Preparamos los datos para la gráfica
        # Simplificamos los nombres de las carpetas para que quepan en la gráfica
        etiquetas = [os.path.basename(c) for c in conteo_por_carpeta.keys()]
        valores = list(conteo_por_carpeta.values())

        plt.figure(figsize=(10, 6)) # Tamaño de la figura
        barras = plt.bar(etiquetas, valores, color='skyblue', edgecolor='black')

        # Añadir títulos y etiquetas
        plt.xlabel('Carpetas', fontsize=12)
        plt.ylabel('Cantidad de archivos JSON', fontsize=12)
        plt.title('Conteo de archivos JSON por carpeta', fontsize=14)
        
        # Rotar los nombres de las carpetas si son muy largos
        plt.xticks(rotation=45, ha='right')
        
        # Añadir el número exacto encima de cada barra
        plt.bar_label(barras, padding=3)

        # Ajustar el diseño para que no se corte nada
        plt.tight_layout()

        # Mostrar la gráfica
        print("\n--> Generando gráfica...")
        plt.show()
        
    else:
        print("No hay archivos para graficar.")

else:
    print(f"ERROR: No encuentro la carpeta '{ruta_base}'.")
    print("Asegúrate de ejecutar el script desde la ubicación correcta.")