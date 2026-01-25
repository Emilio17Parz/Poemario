import os

# =====================================================
# ⚙️ CONFIGURACIÓN
# =====================================================
BASE_DIR = r"C:\Users\calza\Poemario\datasetimg"
# Palabra única para que la IA identifique TU estilo
TRIGGER_WORD = "poemario_style" 
EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp")

def limpiar_texto(texto):
    """Convierte 'bosque_oscuro' en 'bosque oscuro'"""
    return texto.replace("_", " ").replace("-", " ").lower()

def procesar_dataset():
    print(f"🚀 Iniciando etiquetado en: {BASE_DIR}")
    archivos_creados = 0

    for root, dirs, files in os.walk(BASE_DIR):
        # Filtrar solo imágenes
        imagenes = [f for f in files if f.lower().endswith(EXTENSIONES_IMAGEN)]
        
        if not imagenes:
            continue

        # Obtener Concepto y Subconcepto de la ruta
        rel_path = os.path.relpath(root, BASE_DIR)
        partes = rel_path.split(os.sep)
        
        concepto = limpiar_texto(partes[0])
        subconcepto = limpiar_texto(partes[1]) if len(partes) > 1 else ""

        # Crear la cadena de etiquetas (Tags)
        # Formato: Trigger Word, Concepto, Subconcepto
        tags = f"{TRIGGER_WORD}, {concepto}"
        if subconcepto and subconcepto != concepto:
            tags += f", {subconcepto}"

        for img_name in imagenes:
            # Nombre base (sin extensión)
            nombre_base = os.path.splitext(img_name)[0]
            txt_path = os.path.join(root, nombre_base + ".txt")

            # Solo creamos el archivo si no existe
            if not os.path.exists(txt_path):
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(tags)
                archivos_creados += 1

    print(f"--- ✅ Proceso terminado ---")
    print(f"📝 Archivos .txt creados: {archivos_creados}")

if __name__ == "__main__":
    procesar_dataset()