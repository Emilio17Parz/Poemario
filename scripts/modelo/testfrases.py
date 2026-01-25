import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import datetime
import os

# --- CONFIGURACIÓN ---
MODELO_DIR = "./modelo_poemas_neo_balanceado_v3"
ARCHIVO_SALIDA = "test_resultados_frases.txt"

# --- DICCIONARIO DE LÍMITES (La Tijera) ---
LIMITES_LINEAS = {
    "Pareado": 2, "Haiku": 3, "Terceto": 3, "Tanka": 5, 
    "Cuarteta": 4, "Redondilla": 4, "Cuarteto": 4, "Serventesio": 4,
    "Copla": 4, "Seguidilla": 4, "Estrofa alcaica": 4, "Estrofa safica": 4,
    "Limerick": 5, "Lira": 5, "Quintilla": 5,
    "Triolet": 8, "Decima espinela": 10, "Soneto": 14, 
    "Rondeau": 15, "Villanelle": 19,
    "Sestina": 39, "Pantoum": 16,
    "Romance": 20, "Silva": 20, "Verso libre": 15, "Poema en prosa": 10,
    "Zejel": 10, "Moaxaja": 10, "Madrigal": 12, "Ghazal": 14, "Gacela": 14
}

# --- TEMAS: FRASES DE CONTEXTO ---
# Frases complejas para obligar al modelo a narrar una situación específica
TEMAS_FRASES = [
    "el susurro del viento en una casa abandonada",
    "la última lágrima antes de decir adiós",
    "un reloj que camina hacia atrás en el tiempo",
    "la sombra que persigue al caminante solitario",
    "el eco de una risa olvidada en la infancia",
    "dos amantes separándose bajo la lluvia",
    "el misterio que esconde el fondo del mar oscuro",
    "un grito desesperado en el silencio de la noche",
    "la esperanza que nace con el primer rayo de sol",
    "el frío invierno que congela los recuerdos felices",
    "un espejo antiguo que refleja el alma desnuda",
    "la batalla perdida antes de empezar",
    "el aroma del café en una mañana de domingo",
    "las huellas borradas por la arena y el mar",
    "un viaje sin retorno hacia lo desconocido",
    "la nostalgia de un tiempo que no volverá jamás",
    "el fuego que arde sin consumirse en el pecho",
    "la soledad inmensa de una ciudad vacía",
    "un secreto guardado en una caja de música",
    "la belleza efímera de una flor marchita",
    "el instante preciso en que todo cambió para siempre",
    "voces que susurran secretos desde el más allá",
    "la calma tensa después de la tormenta perfecta",
    "un sueño atrapado en una telaraña de plata",
    "la melodía triste de un violín roto"
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
    lineas = texto_generado.split('\n')
    lineas_limpias = [l.strip() for l in lineas if l.strip()]
    
    limite = LIMITES_LINEAS.get(tipo_poema, 14)
    
    if len(lineas_limpias) > limite:
        lineas_limpias = lineas_limpias[:limite]
        cortado = True
    else:
        cortado = False
        
    poema_final = "\n".join(lineas_limpias)
    return poema_final, cortado

def generar_texto(modelo, tokenizer, prompt, device, tipo_poema):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    limite_lineas = LIMITES_LINEAS.get(tipo_poema, 20)
    tokens_estimados = limite_lineas * 16 # Un poco más de margen para frases largas
    max_len = min(350, tokens_estimados + len(inputs["input_ids"][0]) + 50)

    with torch.no_grad():
        output_sequences = modelo.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            do_sample=True,
            temperature=0.85,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    texto_raw = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    if prompt in texto_raw:
        texto_generado = texto_raw.replace(prompt, "").strip()
    else:
        texto_generado = texto_raw[len(prompt):].strip() if len(texto_raw) > len(prompt) else texto_raw

    if "### POEMA:" in texto_generado:
        try:
            texto_generado = texto_generado.split("### POEMA:")[1]
        except IndexError:
            pass
    
    poema_final, fue_cortado = limpiar_y_cortar_poema(texto_generado, tipo_poema)
    return poema_final, fue_cortado

def main():
    print("🚀 CARGANDO MODELO (MODO FRASES)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODELO_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODELO_DIR).to(device)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    resultados = []
    header = f"TEST POÉTICO CON FRASES - {datetime.datetime.now()}\n" + "="*60 + "\n"
    resultados.append(header)
    print(header)

    total = len(REGLAS_ESTRUCTURA)
    for i, (tipo_poema, instruccion) in enumerate(REGLAS_ESTRUCTURA.items(), 1):
        tema_frase = random.choice(TEMAS_FRASES)
        
        prompt = (
            f"TIPO: {tipo_poema}\n"
            f"INSTRUCCION: {instruccion}\n"
            f"TEMA: {tema_frase}\n"
            f"### POEMA:\n"
        )
        
        print(f"[{i}/{total}] Generando {tipo_poema} sobre '{tema_frase[:30]}...' -> ", end="")
        
        poema, cortado = generar_texto(model, tokenizer, prompt, device, tipo_poema)
        
        estado = "[✂️ Cortado]" if cortado else "[✅ Ok]"
        print(estado)
        
        resultado_str = (
            f"\n--- TEST {i}: {tipo_poema.upper()} {estado} ---\n"
            f"TEMA: {tema_frase}\n"
            f"{poema}\n"
            f"{'-'*30}"
        )
        resultados.append(resultado_str)

    with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
        f.write("\n".join(resultados))
        
    print(f"\n✅ ¡TEST CON FRASES COMPLETADO! Resultados en: {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    main()