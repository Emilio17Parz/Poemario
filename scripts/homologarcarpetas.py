import os
import shutil
import unicodedata

DATASET_PATH = "datasets"

def normalizar(nombre):
    # Quitar tildes, espacios, mayúsculas, guiones
    nfkd = unicodedata.normalize("NFKD", nombre)
    s = "".join([c for c in nfkd if not unicodedata.combining(c)])
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    return s.lower()

grupos = {}

for folder in os.listdir(DATASET_PATH):
    ruta = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(ruta):
        key = normalizar(folder)
        if key not in grupos:
            grupos[key] = []
        grupos[key].append(folder)

for key, carpetas in grupos.items():
    if len(carpetas) > 1:
        print(f"\n[+] Carpetas duplicadas detectadas para '{key}': {carpetas}")

        # Elegir la principal (nombre más corto o con guión bajo)
        carpeta_principal = sorted(carpetas, key=lambda x: len(x))[0]
        ruta_principal = os.path.join(DATASET_PATH, carpeta_principal)

        print(f" → Carpeta principal: {carpeta_principal}")

        for carpeta in carpetas:
            if carpeta == carpeta_principal:
                continue

            ruta = os.path.join(DATASET_PATH, carpeta)

            for archivo in os.listdir(ruta):
                src = os.path.join(ruta, archivo)
                dst = os.path.join(ruta_principal, archivo)

                # Si existe un archivo con el mismo nombre, renombrar
                if os.path.exists(dst):
                    nombre, ext = os.path.splitext(archivo)
                    nuevo = f"{nombre}_duplicado{ext}"
                    dst = os.path.join(ruta_principal, nuevo)
                    print(f"   Archivo duplicado detectado: renombrado a {nuevo}")

                shutil.move(src, dst)

            print(f" - Eliminando carpeta vacía: {carpeta}")
            os.rmdir(ruta)

print("\nProceso completado. Carpetas homologadas correctamente.")
