import os
import time
import json
import uuid
import random
import google.generativeai as genai
from datetime import datetime

# ================= CONFIGURACIÓN =================
# --- PON TU API KEY AQUÍ ---
API_KEY = "AIzaSyDQPZPqg2vImRkP14eNuf8HGiVCsS2HtJE"

# Ruta base de tus datasets
BASE_OUTPUT_FOLDER = r"C:\Users\jecal\Poemario\datasets"

# Configuración de Gemini
genai.configure(api_key=API_KEY)
# 'gemini-1.5-flash' es ideal para alto volumen y tareas estructuradas
model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})

# ================= DICCIONARIO MAESTRO DE REGLAS =================
# Mapeamos tus nombres de carpeta a las definiciones poéticas reales.
# He agrupado las definiciones para evitar repetir código, pero todas tus carpetas están cubiertas.

DEFINICIONES = {
    # --- Formas Métricas Clásicas y Estrofas ---
    "soneto": {"desc": "Poema de 14 versos (dos cuartetos y dos tercetos), rima consonante, generalmente endecasílabos."},
    "Cuarteta": {"desc": "Estrofa de 4 versos de arte menor (octosílabos) con rima abab."},
    "Cuarteto": {"desc": "Estrofa de 4 versos de arte mayor (endecasílabos) con rima ABBA."},
    "Redondilla": {"desc": "Estrofa de 4 versos de arte menor (octosílabos) con rima abba."},
    "Serventesio": {"desc": "Estrofa de 4 versos de arte mayor (endecasílabos) con rima ABAB."},
    "Terceto": {"desc": "Estrofa de 3 versos, generalmente endecasílabos."},
    "Terceto encadenado": {"desc": "Serie de tercetos con rima ABA BCB CDC..."},
    "Terceto_encadenado": {"desc": "Serie de tercetos con rima ABA BCB CDC..."},
    "Pareado": {"desc": "Estrofa de 2 versos que riman entre sí."},
    "Decima espinela": {"desc": "Estrofa de 10 versos octosílabos con rima abbaaccddc."},
    "Decima_espinela": {"desc": "Estrofa de 10 versos octosílabos con rima abbaaccddc."},
    "Copla": {"desc": "Estrofa popular de 4 versos de arte menor, rima asonante en pares."},
    "Seguidilla": {"desc": "Estrofa de 4 versos (7-5-7-5), rima asonante en los pares."},
    "Silva": {"desc": "Serie indefinida de versos de 7 y 11 sílabas con rima libre o suelta."},
    "Lira": {"desc": "Estrofa de 5 versos (7a, 11B, 7a, 7b, 11B)."}, # Asumo que 'Estancia' puede referirse a Liras o Estancias de la canción
    "Estancia": {"desc": "Estrofa formada por una combinación de endecasílabos y heptasílabos que se repite a lo largo del poema."},
    
    # --- Formas Fijas Complejas y Repetitivas ---
    "Villanelle": {"desc": "Poema de 19 versos: 5 tercetos y 1 cuarteto. Dos rimas y dos estribillos que se repiten."},
    "Sestina": {"desc": "Poema de 39 versos. 6 estrofas de 6 versos + envión de 3. Las palabras finales de los versos se repiten en orden espiral."},
    "Pantoum": {"desc": "Poema donde el 2º y 4º verso de cada estrofa se convierten en el 1º y 3º de la siguiente."},
    "Rondeau": {"desc": "Forma fija francesa con 15 versos y un estribillo (rentrement) que se repite."},
    "Rondo": {"desc": "Variante del rondeau, poema con repetición cíclica de un tema inicial."},
    "Triolet": {"desc": "Poema de 8 versos donde el 1º se repite en el 4º y 7º, y el 2º en el 8º."},
    "Ghazal": {"desc": "Forma de origen árabe/persa. Dísticos autónomos que terminan con la misma palabra (radif) precedida por la rima (qafiya)."},
    "ghazal": {"desc": "Forma de origen árabe/persa. Dísticos autónomos que terminan con la misma palabra (radif) precedida por la rima (qafiya)."},
    "Gacela": {"desc": "Adaptación española del Ghazal (Lorquiana). Temas de deseo y misticismo."},
    "Gacela_ghazal": {"desc": "Fusión o referencia al estilo del Ghazal clásico."},
    "Moaxaja": {"desc": "Poema estrófico andalusí escrito en árabe clásico o hebreo, que termina con una jarcha en romance o árabe vulgar."},
    "Zejel": {"desc": "Poema estrófico con estribillo, mudanza (3 versos monorrimos) y vuelta."},

    # --- Formas Japonesas y Breves ---
    "Haiku": {"desc": "Poema breve de 3 versos de 5, 7 y 5 sílabas. Captura un instante de la naturaleza (kigo)."},
    "Tanka": {"desc": "Poema japonés de 5 versos: 5-7-5-7-7 sílabas."},
    "Limerick": {"desc": "Poema humorístico de 5 versos con rima AABBA. Ritmo anapéstico muy marcado."},
    "Epigrama": {"desc": "Poema muy breve, agudo, festivo o satírico. Busca el chiste final."},

    # --- Géneros Mayores y Temáticos ---
    "Aventura_epica_heroismo": {"desc": "Narra hazañas. Tono épico, glorioso, objetivo."},
    "Poema epico": {"desc": "Narración extensa de acciones trascendentales para un pueblo. Héroes y dioses."},
    "Poema_epico": {"desc": "Narración extensa de acciones trascendentales para un pueblo. Héroes y dioses."},
    "Poema narrativo": {"desc": "Poema que cuenta una historia con personajes y trama, ritmo marcado."},
    "Poema_narrativo": {"desc": "Poema que cuenta una historia con personajes y trama, ritmo marcado."},
    "Romance": {"desc": "Serie indefinida de versos octosílabos con rima asonante en los pares. Narrativo o lírico."},
    "Balada": {"desc": "Poema narrativo de tono sentimental o legendario, a menudo con estribillo."},
    
    "Desamor_tristeza_perdida": {"desc": "Lírica del dolor, la ausencia y el duelo emocional."},
    "Elegia": {"desc": "Lamento por la muerte de alguien o una desgracia."},
    "Poema elegiaco": {"desc": "Tono triste, de lamento y pérdida."},
    "Poema_elegiaco": {"desc": "Tono triste, de lamento y pérdida."},
    
    "Egloga": {"desc": "Poesía pastoril. Pastores dialogan sobre amores en naturaleza idealizada."},
    "Oda": {"desc": "Alabanza o reflexión elevada sobre un tema, objeto o persona."},
    "Himno": {"desc": "Canto solemne de alabanza a dioses, patria o ideales."},
    
    "Religion_espiritualidad": {"desc": "Mística, conexión divina, fe, dudas espirituales."},
    "Versiculo": {"desc": "Estilo bíblico o de verso largo y solemne, sin rima fija pero con ritmo interno."},
    
    "Sociedad_critica_social": {"desc": "Denuncia de injusticias, política, pobreza o corrupción."},
    "Poema satirico": {"desc": "Burla de vicios o personas. Uso de ironía y sarcasmo."},
    "Poema_satirico": {"desc": "Burla de vicios o personas. Uso de ironía y sarcasmo."},
    
    "Yo_interior_introspeccion": {"desc": "Exploración del yo, la conciencia y los sentimientos íntimos."},
    "Vida_y_existencia": {"desc": "Filosofía, paso del tiempo, sentido de la vida."},

    "Poema didactico": {"desc": "Su fin es enseñar o instruir (fábulas, epístolas morales)."},
    "Poema_didactico": {"desc": "Su fin es enseñar o instruir (fábulas, epístolas morales)."},
    
    "Poema dramatico": {"desc": "Monólogo o diálogo teatral en verso. Conflicto intenso."},
    "Poema_dramatico": {"desc": "Monólogo o diálogo teatral en verso. Conflicto intenso."},

    # --- Formas Experimentales y Visuales ---
    "Acrostico": {"desc": "Las letras iniciales, medias o finales de los versos forman una palabra o frase al leerse verticalmente."},
    "caligrama": {"desc": "La disposición tipográfica de las palabras dibuja el objeto del que se habla."},
    "Poema concreto": {"desc": "Poesía visual. La forma visual es tan importante como el significado."},
    "Poema_concreto": {"desc": "Poesía visual. La forma visual es tan importante como el significado."},
    "Palindromo poetico": {"desc": "El texto puede leerse igual de izquierda a derecha y viceversa (letra a letra o palabra a palabra)."},
    "Palindromo_poetico": {"desc": "El texto puede leerse igual de izquierda a derecha y viceversa (letra a letra o palabra a palabra)."},
    
    "Poema en prosa": {"desc": "Prosa con intensidad lírica, ritmo y figuras poéticas, sin versos."},
    "Poema_en_prosa": {"desc": "Prosa con intensidad lírica, ritmo y figuras poéticas, sin versos."},
    "Verso libre": {"desc": "Sin métrica ni rima fija, guiado por el ritmo del pensamiento."},
    "Verso_libre": {"desc": "Sin métrica ni rima fija, guiado por el ritmo del pensamiento."},

    # --- Otras formas clásicas/raras ---
    "Cancion petrarquista": {"desc": "Composición de varias estancias (combinación de 11 y 7) con un envión final."},
    "Cancion_petrarquista": {"desc": "Composición de varias estancias (combinación de 11 y 7) con un envión final."},
    "Madrigal": {"desc": "Poema breve, amoroso, combinación libre de heptasílabos y endecasílabos."},
    "Estrofa alcaica": {"desc": "Imitación de la estrofa griega (Alceo). Ritmo solemne."},
    "Estrofa_alcaica": {"desc": "Imitación de la estrofa griega (Alceo). Ritmo solemne."},
    "Estrofa safica": {"desc": "Imitación de la estrofa de Safo (tres versos sáficos y un adónico)."},
    "Estrofa_safica": {"desc": "Imitación de la estrofa de Safo (tres versos sáficos y un adónico)."},
    "Poema lirico": {"desc": "Expresión subjetiva de sentimientos."},
    "Poema_lirico": {"desc": "Expresión subjetiva de sentimientos."}
}

# ================= FUNCIONES =================

def generar_lote(categoria_nombre, cantidad):
    # Buscamos la definición. Si no está exacta, usamos una genérica.
    info = DEFINICIONES.get(categoria_nombre, {"desc": "Poema libre temático.", "style": "Libre"})
    
    prompt = f"""
    Actúa como un sistema generador de datos para entrenamiento de LLM.
    
    OBJETIVO:
    Generar una lista JSON de {cantidad} poemas que pertenezcan a la categoría: "{categoria_nombre}".
    
    REGLAS DE LA FORMA/CONTENIDO: 
    {info['desc']}
    
    INSTRUCCIONES CLAVE:
    1. INVENTA una "palabra_clave_ingresada" para cada poema (ej. "Atardecer", "Guerra", "Silencio") que sirva de inspiración.
    2. El poema debe respetar rigurosamente la estructura si es una forma fija (ej. Soneto, Haiku).
    3. Si es un poema visual (Caligrama), escribe el texto e indica entre paréntesis cómo debería verse.
    
    FORMATO DE SALIDA (ESTRICTAMENTE JSON):
    Devuelve una lista de objetos con este esquema EXACTO:
    [
      {{
        "subcategoria": "{categoria_nombre}",
        "poema": {{
          "texto": "Aquí el poema con saltos de línea \\n explícitos",
          "tipo": "{categoria_nombre}",
          "palabra_clave_ingresada": "Palabra inventada"
        }}
      }}
    ]
    """

    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        if isinstance(data, dict): data = [data] # Asegurar lista
        return data
    except Exception as e:
        print(f"Error en generación: {e}")
        return None

def guardar_archivos(lista_datos, carpeta_nombre):
    ruta_dir = os.path.join(BASE_OUTPUT_FOLDER, carpeta_nombre)
    if not os.path.exists(ruta_dir):
        os.makedirs(ruta_dir)
        
    count = 0
    for item in lista_datos:
        # Generar nombre único
        uid = uuid.uuid4().hex[:10]
        fname = f"{carpeta_nombre}_{uid}.json"
        
        with open(os.path.join(ruta_dir, fname), 'w', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False, indent=4)
        count += 1
    return count

def main():
    print("--- GENERADOR POÉTICO MASIVO (74 CATEGORÍAS) ---")
    
    # Obtenemos la lista de carpetas disponibles según tu input
    # (Usamos las claves del diccionario que coinciden con tu lista)
    categorias_disponibles = sorted(list(DEFINICIONES.keys()))
    
    # Menú simple
    for i, cat in enumerate(categorias_disponibles):
        print(f"{i+1}. {cat}")
        
    try:
        idx = int(input("\nSelecciona el número de la categoría: ")) - 1
        categoria_seleccionada = categorias_disponibles[idx]
    except:
        print("Selección inválida.")
        return

    try:
        objetivo = int(input(f"¿Cuántos poemas generar para '{categoria_seleccionada}'? "))
    except:
        return

    print(f"\n---> Generando {objetivo} poemas en: {os.path.join(BASE_OUTPUT_FOLDER, categoria_seleccionada)}")
    
    generados = 0
    lote_size = 5 # Lotes pequeños para asegurar calidad en estructuras complejas
    
    while generados < objetivo:
        falta = objetivo - generados
        pedir = min(lote_size, falta)
        
        print(f"Solicitando lote de {pedir}... (Total: {generados}/{objetivo})")
        
        datos = generar_lote(categoria_seleccionada, pedir)
        
        if datos:
            n = guardar_archivos(datos, categoria_seleccionada)
            generados += n
            time.sleep(5) # Pausa de cortesía API
        else:
            print("Reintentando tras error...")
            time.sleep(5)
            
    print(f"\n¡COMPLETADO! {generados} archivos creados.")

if __name__ == "__main__":
    main()