import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import datetime
import os

# --- CONFIGURACIÓN ---
MODELO_DIR = "./modelo_poemas_neo_balanceado_v3"
ARCHIVO_SALIDA = "test_resultados_cortados.txt"

# --- DICCIONARIO DE LÍMITES (La "Tijera") ---
# Define cuántos versos (líneas) debe tener cada estructura como máximo.
# Si ponemos 0, es libre (o se limita solo por longitud de tokens).
LIMITES_LINEAS = {
    # Cortos y Fijos
    "Pareado": 2, "Haiku": 3, "Terceto": 3, "Tanka": 5, 
    "Cuarteta": 4, "Redondilla": 4, "Cuarteto": 4, "Serventesio": 4,
    "Copla": 4, "Seguidilla": 4, "Estrofa alcaica": 4, "Estrofa safica": 4,
    "Limerick": 5, "Lira": 5, "Quintilla": 5,
    
    # Medios y Fijos
    "Triolet": 8, "Decima espinela": 10, "Soneto": 14, 
    "Rondeau": 15, "Villanelle": 19,
    
    # Largos o Complejos (Pongo un límite de seguridad para que no divague)
    "Sestina": 39, "Pantoum": 16, # Mínimo un pantoum suele ser 3 o 4 estrofas
    
    # Libres (Les ponemos un tope de 16-20 líneas para que no sean eternos)
    "Romance": 20, "Silva": 20, "Verso libre": 15, "Poema en prosa": 10,
    "Zejel": 10, "Moaxaja": 10, "Madrigal": 12, "Ghazal": 14, "Gacela": 14
}

# --- TEMAS RANDOM ---
TEMAS_RANDOM = [
    "el tiempo", "la lluvia", "un espejo", "la soledad", "el caos", 
    "la esperanza", "un gato negro", "la luna llena", "el olvido", 
    "una taza de café", "el mar", "la guerra", "un reloj de arena", 
    "el primer amor", "la muerte", "la primavera", "un bosque antiguo", 
    "el silencio", "la traición", "un sueño lúcido", "la locura", 
    "las estrellas", "un camino de tierra", "el viento", "la nostalgia", 
    "el fuego", "una carta perdida", "el invierno", "la niñez", 
    "un fantasma", "la libertad", "el abismo", "una rosa marchita", 
    "la tecnología", "el universo", "la ira", "la paz", "un robot"
]

REGLAS_ESTRUCTURA = {
    "Pareado": "Escribe un Pareado: 2 versos que rimen entre sí (AA). Medida libre.",
    "Terceto": "Escribe un Terceto: 3 versos de 11 sílabas. Rima ABA.",
    "Cuarteta": "Escribe una Cuarteta: 4 versos de 8 sílabas. Rima cruzada (abab).",
    "Redondilla": "Escribe una Redondilla: 4 versos de 8 sílabas. Rima abrazada (abba).",
    "Cuarteto": "Escribe un Cuarteto: 4 versos de 11 sílabas. Rima abrazada (ABBA).",
    "Serventesio": "Escribe un Serventesio: 4 versos de 11 sílabas. Rima cruzada (ABAB).",
    "Copla": "Escribe una Copla: 4 versos de 8 sílabas. Solo riman los pares (2 y 4). Impares sueltos.",
    "Seguidilla": "Escribe una Seguidilla: 4 versos. Sílabas: 7-5-7-5. Riman los pares en asonante.",
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
    "Villanelle": "Escribe una Villanelle: 19 versos (5 tercetos + 1 cuarteto). Solo dos rimas. Los versos 1 y 3 se repiten alternados.",
    "Sestina": "Escribe una Sestina (o sus 39 versos o un fragmento válido): 6 palabras finales se repiten en todas las estrofas en orden espiral.",
    "Pantoum": "Escribe un Pantoum: Cuartetos donde los versos 2 y 4 de una estrofa se convierten en el 1 y 3 de la siguiente.",
    "Rondeau": "Escribe un Rondeau: 15 versos. Rima aabba aabR aabbaR (R es la primera parte del primer verso usada como estribillo).",
    "Rondo": "Escribe un Rondó: Poema con estructura musical de repetición, similar al Rondeau pero con variaciones libres.",
    "Triolet": "Escribe un Triolet: 8 versos. Esquema ABaAabAB (Las mayúsculas son versos que se repiten idénticos).",
    "Limerick": "Escribe un Limerick: 5 versos de tono humorístico o absurdo. Rima AABBA. Ritmo anapéstico.",
    "Haiku": "Escribe un Haiku: 3 versos. Sílabas 5-7-5. Captura un instante presente de la naturaleza (aquí y ahora). Sin rima.",
    "Tanka": "Escribe un Tanka: 5 versos. Sílabas 5-7-5-7-7. Empieza como imagen natural y termina con emoción profunda.",
    "ghazal": "Escribe un Ghazal (Gacela persa): Dísticos (pareados). Rima monorrima AA, bA, cA... con una palabra final repetida (Radif).",
    "Gacela": "Escribe una Gacela (estilo Lorquiano): Poema de intensidad erótica o mística, inspirada en la forma árabe pero con libertad métrica.",
    "Acrostico": "Escribe un Acróstico: Las letras iniciales de los versos deben leerse verticalmente formando la palabra clave.",
    "caligrama": "Escribe un Caligrama: El texto debe describir visualmente al objeto (Nota: genera el texto sugiriendo la forma).",
    "Palindromo poetico": "Escribe un Palíndromo o texto bifronte: Que pueda leerse igual de izquierda a derecha y viceversa (o frase a frase).",
    "Verso libre": "Escribe en Verso Libre: Sin métrica, ni rima, ni estrofa fija. Sigue el ritmo del pensamiento o la respiración.",
    "Versiculo": "Escribe en Versículos: Versos de gran extensión, ritmo majestuoso o profético (estilo bíblico o Whitman), sin rima.",
}

def limpiar_y_cortar_poema(texto_generado, tipo_poema):
    """
    Limpia el texto de basura y lo corta según el límite de líneas de la estructura.
    """
    # 1. Separar por líneas
    lineas = texto_generado.split('\n')
    
    # 2. Limpieza básica: quitar líneas vacías o muy cortas al inicio/final
    lineas_limpias = [l.strip() for l in lineas if l.strip()]
    
    # 3. Aplicar La Tijera (Límite de líneas)
    limite = LIMITES_LINEAS.get(tipo_poema, 14) # Por defecto 14 si no está en la lista
    
    if len(lineas_limpias) > limite:
        lineas_limpias = lineas_limpias[:limite]
        cortado = True
    else:
        cortado = False
        
    # 4. Reconstruir poema
    poema_final = "\n".join(lineas_limpias)
    
    return poema_final, cortado

def generar_texto(modelo, tokenizer, prompt, device, tipo_poema):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Calculamos un max_length dinámico para no cortar antes de tiempo si es largo
    limite_lineas = LIMITES_LINEAS.get(tipo_poema, 20)
    tokens_estimados = limite_lineas * 15 # Estima 15 tokens por verso
    max_len = min(300, tokens_estimados + len(inputs["input_ids"][0]) + 50)

    with torch.no_grad():
        output_sequences = modelo.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            do_sample=True,
            temperature=0.85,       # Un poco menos caótico
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    texto_raw = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    # Limpiar el prompt del resultado
    if prompt in texto_raw:
        texto_generado = texto_raw.replace(prompt, "").strip()
    else:
        # Fallback por si el decode cambia algo sutilmente
        texto_generado = texto_raw[len(prompt):].strip() if len(texto_raw) > len(prompt) else texto_raw

    # Limpiar basura común del dataset (source: X, ### POEMA, etc)
    if "### POEMA:" in texto_generado:
        texto_generado = texto_generado.split("### POEMA:")[1]
    
    # POST-PROCESAMIENTO (LA TIJERA)
    poema_final, fue_cortado = limpiar_y_cortar_poema(texto_generado, tipo_poema)
    
    return poema_final, fue_cortado

def main():
    print("🚀 CARGANDO MODELO... (Modo Tijera Activado)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODELO_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODELO_DIR).to(device)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    resultados = []
    header = f"TEST POÉTICO ESTRUCTURADO - {datetime.datetime.now()}\n" + "="*60 + "\n"
    resultados.append(header)
    print(header)

    total = len(REGLAS_ESTRUCTURA)
    for i, (tipo_poema, instruccion) in enumerate(REGLAS_ESTRUCTURA.items(), 1):
        tema_random = random.choice(TEMAS_RANDOM)
        
        prompt = (
            f"TIPO: {tipo_poema}\n"
            f"INSTRUCCION: {instruccion}\n"
            f"TEMA: {tema_random}\n"
            f"### POEMA:\n"
        )
        
        print(f"[{i}/{total}] Generando {tipo_poema} ({tema_random})... ", end="")
        
        poema, cortado = generar_texto(model, tokenizer, prompt, device, tipo_poema)
        
        estado = "[✂️ Cortado]" if cortado else "[✅ Ok]"
        print(estado)
        
        resultado_str = (
            f"\n--- TEST {i}: {tipo_poema.upper()} {estado} ---\n"
            f"TEMA: {tema_random}\n"
            f"{poema}\n"
            f"{'-'*30}"
        )
        resultados.append(resultado_str)

    with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
        f.write("\n".join(resultados))
        
    print(f"\n✅ Resultados guardados en: {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    main()