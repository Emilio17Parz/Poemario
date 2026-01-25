import os
import json
import re

# --- CONFIGURACIÓN ---
BASE_DIR = r"C:\Users\calza\Poemario\datasets" # Ajusta si es necesario
# Lista de palabras "basura" que hemos detectado que se cuelan
PALABRAS_PROHIBIDAS = [
    "Yo_interior", "Vida_y_existencia", "Sociedad_critica", 
    "Desamor_tristeza", "Aventura_epica", "Religion_espiritualidad",
    "germano", "ingresada", "subcategoria"
]

def limpiar_texto_poema(texto):
    if not texto: return ""
    
    # 1. Eliminar palabras que contienen guiones bajos (suelen ser variables o nombres de carpetas)
    # Regex: busca palabras que tengan letras y al menos un guion bajo
    texto = re.sub(r'\b\w+_\w+\b', '', texto)

    # 2. Eliminar frases introductorias comunes de modelos
    frases_basura = [
        "Aquí tienes un poema", "El poema es el siguiente:", 
        "Claro, aquí está:", "Titulo:", "Generated:"
    ]
    for frase in frases_basura:
        texto = texto.replace(frase, "")

    # 3. Eliminar las palabras prohibidas explícitas
    for palabra in PALABRAS_PROHIBIDAS:
        texto = texto.replace(palabra, "")

    # 4. Limpieza final de espacios (dobles espacios a uno solo)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def ejecutar_limpieza():
    print(f"🧹 Iniciando limpieza profunda en: {BASE_DIR}")
    archivos_modificados = 0
    
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if not file.endswith(".json"): continue
            filepath = os.path.join(root, file)
            
            try:
                # Leer
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Detectar dónde está el texto
                poema_obj = data.get('poema', {})
                texto_original = poema_obj.get('texto', '')
                
                # Limpiar
                texto_limpio = limpiar_texto_poema(texto_original)
                
                # Si hubo cambios, guardamos
                if texto_original != texto_limpio:
                    data['poema']['texto'] = texto_limpio
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    archivos_modificados += 1
                    print(f"   ✨ Limpiado: {file}")

            except Exception as e:
                print(f"Error en {file}: {e}")

    print("-" * 40)
    print(f"✅ Proceso terminado. Archivos saneados: {archivos_modificados}")

if __name__ == "__main__":
    ejecutar_limpieza()