import torch
import pandas as pd
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# === CONFIGURACIÓN ===
MODEL_PATH = "./modelo_poemas_neo_entrenado"
OUTPUT_TXT = "test_resultados_poemas.txt"
OUTPUT_EXCEL = "test_resultados_poemas.xlsx"

# Diccionario de reglas (mismo que el tuyo)
REGLAS_ESTRUCTURA = {
    "Soneto": "Escribe un Soneto clásico: 14 versos de 11 sílabas. Dos cuartetos (ABBA ABBA) y dos tercetos (CDE CDE o CDC DCD).",
    "Haiku": "Escribe un Haiku: 3 versos. Sílabas 5-7-5. Captura un instante presente de la naturaleza. Sin rima.",
    "Tanka": "Escribe un Tanka: 5 versos. Sílabas 5-7-5-7-7. Empieza como imagen natural y termina con emoción profunda.",
    "Limerick": "Escribe un Limerick: 5 versos de tono humorístico o absurdo. Rima AABBA.",
    "Oda": "Escribe una Oda: Poema de tono elevado y alabanza dirigido a un objeto, persona o concepto abstracto.",
    "Elegia": "Escribe una Elegía: Poema de lamento por la muerte de alguien o la pérdida de algo amado.",
    "Egloga": "Escribe una Égloga: Composición poética del género bucólico, idealizando la vida rústica.",
    "Epigrama": "Escribe un Epigrama: Poema muy breve, agudo, festivo o satírico.",
    "Romance": "Escribe un fragmento de Romance: Tirada de versos de 8 sílabas. Los pares riman en asonante.",
    "Decima espinela": "Escribe una Décima Espinela: 10 versos de 8 sílabas. Rima abbaaccddc.",
    "Redondilla": "Escribe una Redondilla: 4 versos de 8 sílabas. Rima abrazada (abba).",
    "Cuarteta": "Escribe una Cuarteta: 4 versos de 8 sílabas. Rima cruzada (abab).",
    "Cuarteto": "Escribe un Cuarteto: 4 versos de 11 sílabas. Rima abrazada (ABBA).",
    "Serventesio": "Escribe un Serventesio: 4 versos de 11 sílabas. Rima cruzada (ABAB).",
    "Terceto": "Escribe un Terceto: 3 versos de 11 sílabas. Rima ABA.",
    "Terceto encadenado": "Escribe un Terceto Encadenado: Serie de tercetos (ABA BCB CDC...).",
    "Pareado": "Escribe un Pareado: 2 versos que rimen entre sí (AA).",
    "Silva": "Escribe una Silva: Combinación libre de versos de 7 y 11 sílabas con rima consonante.",
    "Copla": "Escribe una Copla: 4 versos de 8 sílabas. Solo riman los pares (2 y 4).",
    "Seguidilla": "Escribe una Seguidilla: 4 versos. Sílabas: 7-5-7-5. Riman los pares en asonante.",
    "Estrofa safica": "Escribe una Estrofa Sáfica: 3 versos de 11 sílabas y uno final de 5 sílabas.",
    "Estrofa alcaica": "Escribe una Estrofa Alcaica: 4 versos (11, 11, 9, 10 sílabas).",
    "Estancia": "Escribe una Estancia: Estrofa formada por versos de 11 y 7 sílabas con patrón fijo.",
    "Balada": "Escribe una Balada: Poema narrativo de tono sentimental dividido en estrofas.",
    "Villanelle": "Escribe una Villanelle: 19 versos con repeticiones alternadas de los versos 1 y 3.",
    "Sestina": "Escribe una Sestina: 6 palabras finales se repiten en todas las estrofas en orden espiral.",
    "Pantoum": "Escribe un Pantoum: Cuartetos donde los versos 2 y 4 se vuelven 1 y 3 de la siguiente.",
    "Rondo": "Escribe un Rondó: Poema con estructura musical de repetición.",
    "Rondeau": "Escribe un Rondeau: 15 versos con estribillo derivado del primer verso.",
    "Triolet": "Escribe un Triolet: 8 versos con esquema ABaAabAB.",
    "Madrigal": "Escribe un Madrigal: Poema breve amoroso, mezcla de 7 y 11 sílabas.",
    "Zejel": "Escribe un Zéjel: Estribillo, mudanza monorrima y verso de vuelta.",
    "Moaxaja": "Escribe una Moaxaja: Poema culto que termina con una Jarcha popular.",
    "Gacela": "Escribe una Gacela: Poema de intensidad erótica o mística de origen árabe.",
    "Cancion petrarquista": "Escribe una Canción Petrarquista: Composición de varias estancias y un envío.",
    "Himno": "Escribe un Himno: Composición solemne destinada al canto religioso o patriótico.",
    "Poema en prosa": "Escribe un Poema en Prosa: Texto en párrafos con ritmo y lenguaje poético.",
    "Verso libre": "Escribe en Verso Libre: Sin métrica, ni rima, ni estrofa fija.",
    "Versiculo": "Escribe en Versículos: Versos largos de ritmo majestuoso sin rima.",
    "Acrostico": "Escribe un Acróstico: Las iniciales de los versos forman una palabra vertical.",
    "Palindromo poetico": "Escribe un Palíndromo: Texto que se lee igual en ambos sentidos.",
    "Poema concreto": "Escribe un Poema Concreto: La disposición visual es parte del mensaje.",
    "Poema narrativo": "Escribe un Poema Narrativo: Cuenta una historia con inicio, nudo y desenlace.",
    "Poema dramatico": "Escribe un Poema Dramático: Versos concebidos para ser representados.",
    "Poema lirico": "Escribe un Poema Lírico: Expresión subjetiva de sentimientos íntimos.",
    "Poema elegiaco": "Escribe un Poema Elegíaco: Enfocado en el dolor y la nostalgia.",
    "Poema epico": "Escribe un Poema Épico: Narración de hazañas de héroes legendarios.",
    "Poema satirico": "Escribe un Poema Satírico: Usa la burla para criticar vicios.",
    "Poema didactico": "Escribe un Poema Didáctico: Su fin principal es enseñar o instruir."
}

# Límites de líneas para el post-procesado
LIMITES_LINEAS = {
    "pareado": 2, "terceto": 3, "cuarteta": 4, "redondilla": 4, "cuarteto": 4,
    "serventesio": 4, "copla": 4, "seguidilla": 4, "haiku": 3, "tanka": 5,
    "limerick": 5, "soneto": 14, "decima espinela": 10, "lira": 5,
    "triolet": 8, "villanelle": 19, "rondeau": 15
}

# Palabras aleatorias para testear
KEYWORDS_TEST = [
    "laberinto", "neón", "olvido", "mariposa", "acero", "crepúsculo", 
    "algoritmo", "espejo", "silencio", "viento", "ceniza", "galaxia",
    "reloj", "sombra", "invierno", "fuego", "raíces", "mármol"
]

def post_procesar(texto, tipo):
    lineas = [l.strip() for l in texto.split('\n') if l.strip()]
    limite = LIMITES_LINEAS.get(tipo.lower(), 40) # 40 por defecto para largos
    return "\n".join(lineas[:limite])

def ejecutar_test_masivo():
    print(f"🚀 Iniciando Test Masivo en {MODEL_PATH}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resultados = []
    
    # Abrir archivo TXT para ir escribiendo en tiempo real
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f_txt:
        f_txt.write(f"TEST DE GENERACIÓN POÉTICA - {datetime.now()}\n")
        f_txt.write("="*50 + "\n\n")

        for i, (tipo, instruccion) in enumerate(REGLAS_ESTRUCTURA.items(), 1):
            keyword = random.choice(KEYWORDS_TEST)
            print(f"[{i}/49] Generando {tipo} sobre '{keyword}'...")

            prompt = f"TIPO: {tipo}\nINSTRUCCION: {instruccion}\nTEMA: {keyword}\n### POEMA:\n"
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    repetition_penalty=1.5,
                    eos_token_id=tokenizer.eos_token_id
                )

            full_text = tokenizer.decode(output[0], skip_special_tokens=True)
            poema_crudo = full_text.split("### POEMA:")[-1].strip()
            poema_final = post_procesar(poema_crudo, tipo)

            # Guardar en lista para Excel
            resultados.append({
                "ID": i,
                "Tipo": tipo,
                "Keyword": keyword,
                "Poema": poema_final
            })

            # Escribir en TXT
            f_txt.write(f"--- TEST {i}: {tipo.upper()} ---\n")
            f_txt.write(f"TEMA: {keyword}\n")
            f_txt.write(f"{poema_final}\n\n")

    # Guardar en Excel
    df = pd.DataFrame(resultados)
    df.to_excel(OUTPUT_EXCEL, index=False)
    
    print(f"\n✅ Test finalizado con éxito.")
    print(f"📂 Resultados guardados en: \n - {OUTPUT_TXT}\n - {OUTPUT_EXCEL}")

if __name__ == "__main__":
    ejecutar_test_masivo()