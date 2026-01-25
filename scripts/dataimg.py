from icrawler.builtin import BingImageCrawler
import os
import time

# =====================================================
# ⚙️ CONFIGURACIÓN
# =====================================================
BASE_DIR = r"C:\Users\calza\Poemario\datasetimg"
CANTIDAD_POR_CARPETA = 800
UMBRAL_EXISTENTES = 750 
TIPO_IMAGEN = "photo"

MODIFICADORES = [
    "", " aesthetic", " fotografia artistica", " wallpaper 4k", 
    " fondo de pantalla", " high resolution", " concept art", 
    " pinterest style", " cinematografico", " realismo", 
    " abstracto", " oscuro", " iluminacion dramatica"
]

# =====================================================
# 🛠️ HELPER: CONTAR IMÁGENES
# =====================================================
def contar_imagenes(ruta):
    try:
        return len([
            f for f in os.listdir(ruta)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
    except Exception:
        return 0

def limpiar_nombre_para_query(nombre_carpeta):
    return nombre_carpeta.replace("_", " ").replace("-", " ")

# =====================================================
# 🚀 FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# =====================================================
def procesar_carpeta_existente(ruta_completa):
    nombre_carpeta = os.path.basename(ruta_completa)
    tema_base = limpiar_nombre_para_query(nombre_carpeta)
    
    for mod in MODIFICADORES:
        actuales = contar_imagenes(ruta_completa)
        
        if actuales >= UMBRAL_EXISTENTES:
            print(f"✅ Meta alcanzada para '{nombre_carpeta}' ({actuales} imgs).")
            return 

        cantidad_faltante = CANTIDAD_POR_CARPETA - actuales
        query_final = f"{tema_base}{mod}"
        
        print(f"\n🌊 Procesando: [{nombre_carpeta}]")
        print(f"   🔍 Query actual: '{query_final}'")
        print(f"   📊 Estado: {actuales}/{CANTIDAD_POR_CARPETA}. Faltan {cantidad_faltante}.")

        crawler = BingImageCrawler(
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4, 
            storage={"root_dir": ruta_completa},
            log_level='ERROR' 
        )
        
        try:
            crawler.crawl(
                keyword=query_final,
                max_num=cantidad_faltante,
                filters={"type": TIPO_IMAGEN}, 
                file_idx_offset='auto',
                overwrite=False
            )
        except Exception as e:
            print(f"⚠️ Error leve en búsqueda '{query_final}': {e}")
            
        time.sleep(1)

    finales = contar_imagenes(ruta_completa)
    print(f"🏁 Se agotaron las búsquedas para '{nombre_carpeta}'. Total conseguido: {finales}")

# =====================================================
# 🏁 EJECUCIÓN CON PRIORIDAD (Carpetas más vacías primero)
# =====================================================
if __name__ == "__main__":
    print(f"--- 🚀 INICIANDO DESCARGA POR PRIORIDAD (Menos imágenes primero) ---")
    
    # 1. Recopilar todas las subcarpetas
    lista_carpetas = []
    for root, dirs, files in os.walk(BASE_DIR):
        for carpeta in dirs:
            if carpeta.startswith("."): continue
            ruta_abs = os.path.join(root, carpeta)
            # Guardamos la ruta y la cantidad actual de imágenes
            lista_carpetas.append({
                "ruta": ruta_abs,
                "cantidad": contar_imagenes(ruta_abs)
            })

    # 2. Ordenar la lista por el valor de "cantidad" (de menor a mayor)
    lista_carpetas_ordenada = sorted(lista_carpetas, key=lambda x: x['cantidad'])

    print(f"📂 Total de carpetas encontradas: {len(lista_carpetas_ordenada)}\n")

    # 3. Procesar en orden
    for item in lista_carpetas_ordenada:
        procesar_carpeta_existente(item["ruta"])
        
        print("⏳ Pausa entre carpetas (2s)...")
        time.sleep(2)

    print(f"\n🎉 ¡Todo listo! Dataset actualizado por orden de necesidad.")