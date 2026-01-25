import os
import imagesize

DIR = r"C:\Users\calza\Poemario\saved_models_final\stable\img"
print("🔍 Buscando imágenes con dimensiones inválidas (0px)...")

eliminados = 0
for root, _, files in os.walk(DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, f)
            try:
                w, h = imagesize.get(path)
                if w <= 0 or h <= 0:
                    raise ValueError(f"Dimensiones inválidas: {w}x{h}")
            except Exception as e:
                print(f"❌ Eliminando imagen con error de tamaño: {f} ({e})")
                os.remove(path)
                # Borrar txt asociado
                txt = os.path.splitext(path)[0] + ".txt"
                if os.path.exists(txt): os.remove(txt)
                eliminados += 1

print(f"\n✅ Limpieza terminada. Archivos eliminados por tamaño 0: {eliminados}")