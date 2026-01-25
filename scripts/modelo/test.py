import torch
import sys
# CAMBIO IMPORTANTE: Usamos AutoModel y AutoTokenizer para compatibilidad total
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ruta donde se guardó tu modelo entrenado (GPT-Neo)
MODEL_PATH = "./modelo_poemas_neo_entrenado"

# === 1. CONFIGURACIÓN DE LÍMITES (La Tijera) ===
LIMITES_LINEAS = {
    "pareado": 2, "terceto": 3, "cuarteta": 4, "redondilla": 4,
    "cuarteto": 4, "serventesio": 4, "copla": 4, "seguidilla": 4,
    "haiku": 3, "tanka": 5, "limerick": 5, "soneto": 14,
    "decima espinela": 10, "lira": 5, "estrofa alcaica": 4, "estrofa safica": 4
}

# === 2. DICCIONARIO DE REGLAS ===
# (Debe ser idéntico al del entrenamiento para que la instrucción coincida)
REGLAS_ESTRUCTURA = {
      # --- Estructuras Básicas ---
    "Pareado": "Escribe un Pareado: 2 versos que rimen entre sí (AA). Medida libre.",
    "Terceto": "Escribe un Terceto: 3 versos de 11 sílabas. Rima ABA.",
    "Terceto encadenado": "Escribe un Terceto Encadenado: Serie de tercetos. La rima del medio del primero es la rima de las puntas del siguiente (ABA BCB CDC...).",
    "Cuarteta": "Escribe una Cuarteta: 4 versos de 8 sílabas. Rima cruzada (abab).",
    "Redondilla": "Escribe una Redondilla: 4 versos de 8 sílabas. Rima abrazada (abba).",
    "Cuarteto": "Escribe un Cuarteto: 4 versos de 11 sílabas. Rima abrazada (ABBA).",
    "Serventesio": "Escribe un Serventesio: 4 versos de 11 sílabas. Rima cruzada (ABAB).",
    "Copla": "Escribe una Copla: 4 versos de 8 sílabas. Solo riman los pares (2 y 4). Impares sueltos.",
    "Seguidilla": "Escribe una Seguidilla: 4 versos. Sílabas: 7-5-7-5. Riman los pares en asonante.",

    # --- Estructuras Fijas ---
    "soneto": "Escribe un Soneto clásico: 14 versos de 11 sílabas. Dos cuartetos (ABBA ABBA) y dos tercetos (CDE CDE o CDC DCD).",
    "Decima espinela": "Escribe una Décima Espinela: 10 versos de 8 sílabas. Rima abbaaccddc. Pausa fuerte obligatoria tras el verso 4.",
    "Romance": "Escribe un fragmento de Romance: Tirada de versos de 8 sílabas. Los pares riman en asonante, impares sueltos.",
    "Silva": "Escribe una Silva: Combinación libre de versos de 7 y 11 sílabas. Rima consonante a gusto del poeta.",
    "Estancia": "Escribe una Estancia: Estrofa formada por versos de 11 y 7 sílabas con rima consonante. Fija un patrón y síguelo.",
    "Cancion petrarquista": "Escribe una Canción Petrarquista: Composición solemne de varias estancias (mezcla de 11 y 7 sílabas) y un envío final.",
    "Madrigal": "Escribe un Madrigal: Poema breve de tema amoroso e idílico. Combinación libre de 7 y 11 sílabas. Rima consonante.",
    "Estrofa alcaica": "Escribe una Estrofa Alcaica (adaptación española): 4 versos. Los dos primeros endecasílabos, el tercero eneasílabo y el último decasílabo.",
    "Estrofa safica": "Escribe una Estrofa Sáfica: 3 versos de 11 sílabas y uno final de 5 sílabas (adónico).",
    "Zejel": "Escribe un Zéjel: Estribillo inicial, mudanza de 3 versos monorrimos y un verso de vuelta que rima con el estribillo.",
    "Moaxaja": "Escribe una Moaxaja: Poema culto (estilo hebreo/árabe andalusí) que termina con una Jarcha (estrofa breve en lenguaje popular).",
    
    # --- Formas Repetitivas ---
    "Villanelle": "Escribe una Villanelle: 19 versos (5 tercetos + 1 cuarteto). Solo dos rimas. Los versos 1 y 3 se repiten alternados.",
    "Sestina": "Escribe una Sestina (o sus 39 versos o un fragmento válido): 6 palabras finales se repiten en todas las estrofas en orden espiral.",
    "Pantoum": "Escribe un Pantoum: Cuartetos donde los versos 2 y 4 de una estrofa se convierten en el 1 y 3 de la siguiente.",
    "Rondeau": "Escribe un Rondeau: 15 versos. Rima aabba aabR aabbaR (R es la primera parte del primer verso usada como estribillo).",
    "Rondo": "Escribe un Rondó: Poema con estructura musical de repetición, similar al Rondeau pero con variaciones libres.",
    "Triolet": "Escribe un Triolet: 8 versos. Esquema ABaAabAB (Las mayúsculas son versos que se repiten idénticos).",
    "Limerick": "Escribe un Limerick: 5 versos de tono humorístico o absurdo. Rima AABBA. Ritmo anapéstico.",

    # --- Exóticas y Orientales ---
    "Haiku": "Escribe un Haiku: 3 versos. Sílabas 5-7-5. Captura un instante presente de la naturaleza (aquí y ahora). Sin rima.",
    "Tanka": "Escribe un Tanka: 5 versos. Sílabas 5-7-5-7-7. Empieza como imagen natural y termina con emoción profunda.",
    "ghazal": "Escribe un Ghazal (Gacela persa): Dísticos (pareados). Rima monorrima AA, bA, cA... con una palabra final repetida (Radif).",
    "Gacela": "Escribe una Gacela (estilo Lorquiano): Poema de intensidad erótica o mística, inspirada en la forma árabe pero con libertad métrica.",
    "Gacela_ghazal": "Escribe un Ghazal estricto: Mantén la estructura Qafiya (rima) y Radif (estribillo) al final de los versos pares.",

    # --- Visuales y Experimentales ---
    "Acrostico": "Escribe un Acróstico: Las letras iniciales de los versos deben leerse verticalmente formando la palabra clave.",
    "caligrama": "Escribe un Caligrama: El texto debe describir visualmente al objeto (Nota: genera el texto sugiriendo la forma).",
    "Poema concreto": "Escribe un Poema Concreto: Donde la disposición visual de las palabras es tan importante como su significado.",
    "Palindromo poetico": "Escribe un Palíndromo o texto bifronte: Que pueda leerse igual de izquierda a derecha y viceversa (o frase a frase).",
    "Verso libre": "Escribe en Verso Libre: Sin métrica, ni rima, ni estrofa fija. Sigue el ritmo del pensamiento o la respiración.",
    "Versiculo": "Escribe en Versículos: Versos de gran extensión, ritmo majestuoso o profético (estilo bíblico o Whitman), sin rima.",

    # --- Géneros (El propósito) ---
    "Balada": "Escribe una Balada: Poema narrativo de tono sentimental o legendario, dividido en estrofas iguales.",
    "Oda": "Escribe una Oda: Poema de tono elevado y alabanza dirigido a un objeto, persona o concepto abstracto.",
    "Elegia": "Escribe una Elegía: Poema de lamento por la muerte de alguien o la pérdida de algo amado.",
    "Poema elegiaco": "Escribe un Poema Elegíaco: Similar a la elegía, enfocado en el dolor, la nostalgia y la fugacidad de la vida.",
    "Himno": "Escribe un Himno: Composición solemne, destinada al canto, que expresa sentimientos patrióticos o religiosos.",
    "Epigrama": "Escribe un Epigrama: Poema muy breve, agudo, festivo o satírico.",
    "Egloga": "Escribe una Égloga: Composición poética del género bucólico, idealizando la vida rústica y de pastores.",
    "Poema epico": "Escribe un fragmento de Poema Épico: Narración en verso de hazañas de héroes legendarios.",
    "Poema narrativo": "Escribe un Poema Narrativo: Cuenta una historia completa (inicio, nudo, desenlace) usando versos.",
    "Poema satirico": "Escribe un Poema Satírico: Usa la ironía y la burla para criticar vicios o defectos sociales/individuales.",
    "Poema dramatico": "Escribe un Poema Dramático: Texto escrito en verso concebido para ser representado (diálogo o monólogo teatral).",
    "Poema en prosa": "Escribe un Poema en Prosa: Texto en párrafos (no versos) pero con la intensidad, ritmo e imágenes de la poesía.",
    "Poema lirico": "Escribe un Poema Lírico: Centrado en la expresión subjetiva de los sentimientos íntimos del poeta.",
    "Poema didactico": "Escribe un Poema Didáctico: Su fin principal es enseñar, instruir o divulgar conocimientos artísticos o científicos.",

    # --- Temáticas (Carpetas de tema) ---
    "Aventura_epica_heroismo": "Escribe un poema centrado en la Aventura, la Épica y el Heroísmo.",
    "Desamor_tristeza_perdida": "Escribe un poema profundo sobre Desamor, Tristeza y Pérdida.",
    "Religion_espiritualidad": "Escribe un poema místico sobre Religión y Espiritualidad.",
    "Sociedad_critica_social": "Escribe un poema de denuncia sobre la Sociedad y Crítica Social.",
    "Vida_y_existencia": "Escribe un poema filosófico sobre la Vida y la Existencia humana.",
    "Yo_interior_introspeccion": "Escribe un poema psicológico sobre el Yo Interior y la Introspección."
}

def post_procesar_poema(texto, tipo):
    lineas = [l.strip() for l in texto.split('\n') if l.strip()]
    tipo_norm = tipo.lower()
    limite = LIMITES_LINEAS.get(tipo_norm, None)
    if limite and len(lineas) > limite:
        lineas = lineas[:limite]
    return "\n".join(lineas)

def mostrar_menu():
    opciones = list(REGLAS_ESTRUCTURA.keys())
    opciones.sort()
    
    print("\n" + "="*40)
    print("      MENÚ DE GENERACIÓN POÉTICA")
    print("="*40)
    
    mitad = (len(opciones) + 1) // 2
    for i in range(mitad):
        op1 = f"{i+1}. {opciones[i]}"
        idx2 = i + mitad
        if idx2 < len(opciones):
            op2 = f"{idx2+1}. {opciones[idx2]}"
            print(f"{op1:<35} | {op2}")
        else:
            print(f"{op1}")
            
    print("="*40)
    print("0. Salir")
    
    while True:
        try:
            seleccion = int(input("\n>>> Elige un número: "))
            if seleccion == 0: return None
            if 1 <= seleccion <= len(opciones): return opciones[seleccion - 1]
            print("Número inválido.")
        except ValueError:
            print("Por favor, ingresa un número.")

def generar_poema(tipo, keyword):
    print(f"\n⚙️ Cargando modelo desde {MODEL_PATH}...")
    try:
        # AQUÍ ESTÁ EL ARREGLO PRINCIPAL:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error crítico cargando modelo: {e}")
        return

    # Si no hay pad token, usar EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Preparar Prompt EXACTAMENTE como se entrenó
    estructura = REGLAS_ESTRUCTURA.get(tipo, f"Escribe un poema de tipo: {tipo}")
    
    # FORMATO NUEVO (Compatible con el entrenamiento)
    prompt = (
        f"TIPO: {tipo}\n"
        f"INSTRUCCION: {estructura}\n"
        f"TEMA: {keyword}\n"
        f"### POEMA:\n"
    )

    print(f"🖋️  Escribiendo {tipo} sobre '{keyword}'...")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Generación
# Generación Ajustada
    output_sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,       # Bajamos de 150 a 100 (un Haiku es corto)
        do_sample=True,
        top_k=50,
        top_p=0.95,               # Subimos un poco para dar más variedad
        temperature=0.7,          # Bajamos temperatura para que sea menos "loco"
        repetition_penalty=1.5,   # SUBIMOS FUERTE: Esto castiga al modelo si repite frases
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    # Cortar el prompt para quedarse solo con el poema nuevo
    # Buscamos la marca "### POEMA:" y tomamos lo que sigue
    if "### POEMA:" in generated_text:
        partes = generated_text.split("### POEMA:")
        texto_poema = partes[-1].strip() # Tomamos la última parte
    else:
        # Fallback por si el modelo se come la etiqueta
        texto_poema = generated_text.replace(prompt, "").strip()

    print("\n" + "-" * 30)
    print(f"✨ {tipo.upper()} GENERADO:\n")
    
    # Aplicar tijera
    poema_final = post_procesar_poema(texto_poema, tipo)
    print(poema_final)
    
    print("-" * 30 + "\n")

if __name__ == "__main__":
    while True:
        tipo_elegido = mostrar_menu()
        if tipo_elegido is None:
            print("¡Hasta luego, poeta!")
            break
            
        palabra_clave = input(f"¿Sobre qué quieres que trate tu {tipo_elegido}? (Keyword): ")
        if not palabra_clave:
            palabra_clave = "vida"
            
        generar_poema(tipo_elegido, palabra_clave)
        
        input("Presiona Enter para volver al menú...")