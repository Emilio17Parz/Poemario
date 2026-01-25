import os
import glob
from collections import Counter
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
# Ruta relativa modificada para buscar imágenes
ruta_base = 'datasetimg'

# Definimos qué extensiones consideramos "imágenes"
extensiones_imagenes = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']

# Verificar que la carpeta exista
if os.path.exists(ruta_base):
    print(f"--> Buscando dentro de: {os.path.abspath(ruta_base)}")
    
    # Búsqueda recursiva para todas las extensiones definidas
    archivos_encontrados = []
    for ext in extensiones_imagenes:
        patron = os.path.join(ruta_base, '**', ext)
        # Extendemos la lista principal con los resultados de cada extensión
        archivos_encontrados.extend(glob.glob(patron, recursive=True))

    total_archivos = len(archivos_encontrados)
    print(f"---------------------------------------")
    print(f"RESULTADO TOTAL: Se encontraron {total_archivos} imágenes.")
    print(f"---------------------------------------")

    if total_archivos > 0:
        # --- PASO 1: PROCESAMIENTO DE DATOS ---
        # Extraemos solo la ruta del directorio de cada archivo encontrado
        carpetas = [os.path.dirname(archivo) for archivo in archivos_encontrados]
        
        # Usamos Counter para contar cuántas veces se repite cada carpeta
        conteo_por_carpeta = Counter(carpetas)

        # Imprimimos el desglose en texto
        print("\nDesglose por carpeta:")
        for carpeta, cantidad in conteo_por_carpeta.items():
            # Mostramos la ruta relativa para que sea más legible
            nombre_corto = os.path.relpath(carpeta, start=os.getcwd())
            print(f" - {nombre_corto}: {cantidad} imágenes")

        # --- PASO 2: GENERACIÓN DE LA GRÁFICA ---
        # Preparamos los datos para la gráfica
        # Simplificamos los nombres de las carpetas para que quepan en la gráfica
        etiquetas = [os.path.basename(c) for c in conteo_por_carpeta.keys()]
        valores = list(conteo_por_carpeta.values())

        plt.figure(figsize=(10, 6)) # Tamaño de la figura
        barras = plt.bar(etiquetas, valores, color='lightgreen', edgecolor='black')

        # Añadir títulos y etiquetas (Actualizados a Imágenes)
        plt.xlabel('Carpetas', fontsize=12)
        plt.ylabel('Cantidad de imágenes', fontsize=12)
        plt.title('Conteo de imágenes por carpeta', fontsize=14)
        
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
        print(f"No se encontraron imágenes en '{ruta_base}'.")

else:
    print(f"ERROR: No encuentro la carpeta '{ruta_base}'.")
    print("Asegúrate de que la carpeta 'datasetimg' exista en la ubicación del script.")