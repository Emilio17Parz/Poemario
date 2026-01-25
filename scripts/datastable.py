import os
import shutil

# =====================================================
# ⚙️ CONFIGURACIÓN
# =====================================================
# Tu dataset principal con las 542k imágenes
ORIGEN = r"C:\Users\calza\Poemario\datasetimg"
# La carpeta que pusiste en Kohya_ss
DESTINO_BASE = r"C:\Users\calza\Poemario\saved_models_final\stable\img"

IMAGENES_POR_CARPETA = 50  # Cantidad moderada para alta calidad
REPETICIONES = 20          # Veces que la IA verá cada foto por época
TRIGGER = "poemario_style" # Tu palabra de activación

EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp")

# =====================================================
# 🚀 PROCESO DE CURACIÓN
# =====================================================
def curar_dataset():
    print(f"🧹 Iniciando curación de dataset para LoRA...")
    os.makedirs(DESTINO_BASE, exist_ok=True)
    
    carpetas_procesadas = 0
    archivos_copiados = 0

    # Recorrer subcarpetas de conceptos
    for root, dirs, files in os.walk(ORIGEN):
        # Filtrar solo imágenes
        imagenes = [f for f in files if f.lower().endswith(EXTENSIONES)]
        
        if not imagenes:
            continue

        # Nombre de la carpeta actual (el concepto)
        nombre_concepto = os.path.basename(root).replace(" ", "_")
        
        # Crear nombre de carpeta para Kohya: "20_poemario_style concepto"
        nombre_dest = f"{REPETICIONES}_{TRIGGER} {nombre_concepto}"
        ruta_dest_final = os.path.join(DESTINO_BASE, nombre_dest)
        os.makedirs(ruta_dest_final, exist_ok=True)

        # Tomar las primeras N imágenes
        seleccion = imagenes[:IMAGENES_POR_CARPETA]
        
        for img_name in seleccion:
            # Ruta origen
            img_old = os.path.join(root, img_name)
            txt_old = os.path.join(root, os.path.splitext(img_name)[0] + ".txt")

            # Copiar imagen
            shutil.copy2(img_old, os.path.join(ruta_dest_final, img_name))
            
            # Copiar el .txt correspondiente (si existe)
            if os.path.exists(txt_old):
                shutil.copy2(txt_old, os.path.join(ruta_dest_final, os.path.basename(txt_old)))
            
            archivos_copiados += 1
            
        carpetas_procesadas += 1
        print(f"✅ Procesada: {nombre_concepto} ({len(seleccion)} imgs)")

    print(f"\n--- 🎉 ¡Curación completada! ---")
    print(f"📂 Carpetas creadas: {carpetas_procesadas}")
    print(f"🖼️ Total archivos copiados: {archivos_copiados}")

if __name__ == "__main__":
    curar_dataset()