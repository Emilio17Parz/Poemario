import os
import torch
import numpy as np
import tensorflow as tf
import joblib
import random
import gc
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. CONFIGURACIÓN ---
BASE_DIR = r"C:\Users\calza\Poemario"
OUTPUT_DIR = os.path.join(BASE_DIR, "galeria_extreme_50")

# Rutas Modelos
GAN_PATH = os.path.join(BASE_DIR, r"saved_models_final\gen_final.keras")
ENCODER_PATH = os.path.join(BASE_DIR, r"saved_models_final\encoder_con.pkl")
LORA_PATH = os.path.join(BASE_DIR, r"saved_models_final\stable\poemario.safetensors")
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
POEM_MODEL_DIR = os.path.join(BASE_DIR, "modelo_poemas_neo_balanceado_v3")

# Configuración GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e: print(e)

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# --- 2. LISTA "EXTREMA" (50 Conceptos Únicos) ---
# Diseñada para probar los límites de la imaginación de la IA
TEMAS_EXTREMOS = [
    # --- Surrealismo y Sueños ---
    "relojes derritiendose en el desierto", "lluvia de paraguas negros", 
    "un ojo gigante en el cielo", "escaleras que suben a la nada", 
    "el ajedrez infinito", "peces nadando en el aire",
    
    # --- Ciencia Ficción y Cyberpunk ---
    "ciborg meditando en un templo", "autopista de datos neon", 
    "la ultima bateria de la tierra", "ciudad flotante entre nubes toxicas", 
    "android soñando con ovejas electricas", "mercado nocturno en marte",
    
    # --- Terror y Oscuridad ---
    "catedral hecha de huesos", "el susurro que sale del espejo", 
    "bosque de arboles con manos", "la sombra que tiene vida propia", 
    "un payaso triste en un hospital abandonado", "el rey de las ratas",
    
    # --- Naturaleza Exótica ---
    "volcan erupcionando hielo azul", "bosque de cristales gigantes", 
    "bioluminiscencia en la playa negra", "aurora boreal sobre piramides", 
    "tornado de fuego", "jardin de plantas carnivoras gigantes",
    
    # --- Conceptos Abstractos y Filosóficos ---
    "la geometria sagrada del alma", "el eco del vacio", 
    "la paradoja de schrodinger", "el sonido del silencio", 
    "nostalgia de un tiempo que no existio", "el caos cuantico",
    
    # --- Objetos Específicos ---
    "una maquina de escribir fantasma", "un barco en una botella rota", 
    "mascara de oro llorando sangre", "una biblioteca laberintica", 
    "trebol de cinco hojas de metal",
    
    # --- Escenarios Épicos ---
    "batalla de angeles caidos", "el fin del universo", 
    "nacimiento de una estrella", "ruinas de una civilizacion acuatica", 
    "el trono del rey olvidado",
    
    # --- Emociones Complejas ---
    "la soledad de un dios", "la furia de la tormenta", 
    "esperanza en una caja de pandora", "el primer beso del apocalipsis",
    
    # --- Rellenos Creativos ---
    "un gato hecho de humo", "mariposas mecanicas oxidadas", 
    "el vals de los fantasmas", "arquitectura imposible"
]

# Aseguramos que haya exactamente 50 (o cortamos si me pasé escribiendo)
TEMAS_EXTREMOS = TEMAS_EXTREMOS[:50] 

# Reglas de estructura (Diccionario completo)
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
}

# Límites de líneas para tijera
LIMITES_LINEAS = {
    "Pareado": 2, "Haiku": 3, "Terceto": 3, "Tanka": 5, 
    "Cuarteta": 4, "Redondilla": 4, "Cuarteto": 4, "Serventesio": 4,
    "Copla": 4, "Seguidilla": 4, "Estrofa alcaica": 4, "Estrofa safica": 4,
    "Limerick": 5, "Lira": 5, "Quintilla": 5,
    "Triolet": 8, "Decima espinela": 10, "Soneto": 14, 
    "Rondeau": 15, "Villanelle": 19,
    # Límites visuales para los largos
    "Sestina": 12, "Pantoum": 12, "Romance": 12, "Silva": 12, 
    "Verso libre": 10, "Poema en prosa": 8, "Zejel": 10, "Moaxaja": 10, 
    "Madrigal": 10, "Ghazal": 10, "Gacela": 10, "Balada": 12, "Oda": 12, 
    "Elegia": 12, "Himno": 12
}

# --- 3. CLASE POETA (Texto con Tijera) ---
class GeneradorTexto:
    def __init__(self):
        print("📖 [1/2] Cargando Poeta...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(POEM_MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(POEM_MODEL_DIR).to(self.device)

    def limpiar_y_cortar(self, texto, tipo_poema):
        lineas = [l.strip() for l in texto.split('\n') if l.strip()]
        limite = LIMITES_LINEAS.get(tipo_poema, 12)
        if len(lineas) > limite:
            lineas = lineas[:limite]
        return "\n".join(lineas)

    def generar(self, tema, tipo_poema, instruccion):
        prompt = f"TIPO: {tipo_poema}\nINSTRUCCION: {instruccion}\nTEMA: {tema}\n### POEMA:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        limite_lineas = LIMITES_LINEAS.get(tipo_poema, 14)
        max_len = min(350, limite_lineas * 20 + len(inputs["input_ids"][0]))

        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_len, 
                do_sample=True, temperature=0.9, top_p=0.92, repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        raw = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "### POEMA:" in raw: texto = raw.split("### POEMA:")[1].strip()
        else: texto = raw.replace(prompt, "").strip()
            
        return self.limpiar_y_cortar(texto, tipo_poema)

    def liberar(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        print("🗑️ Poeta liberado.")

# --- 4. CLASE PINCEL (Imagen Imaginativa) ---
class GeneradorImagen:
    def __init__(self):
        print("🎨 [2/2] Cargando Pincel (GAN + SD)...")
        self.gan = tf.keras.models.load_model(GAN_PATH, compile=False)
        self.le = joblib.load(ENCODER_PATH)
        self.clases_gan = list(self.le.classes_)
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.load_lora_weights(LORA_PATH)

    def crear_hibrido(self, tema):
        # Fallback inteligente: buscar palabras clave
        tema_limpio = tema.strip().lower()
        
        # Intenta encontrar alguna palabra del tema que coincida con una clase GAN
        # Ej: "relojes derritiendose" -> busca si existe "reloj" en GAN
        clase_gan = None
        for palabra in tema_limpio.split():
             match = next((c for c in self.clases_gan if palabra in c.lower()), None)
             if match:
                 clase_gan = match
                 break
        
        if not clase_gan:
            clase_gan = random.choice(self.clases_gan)
            # print(f"      🔸 Base aleatoria GAN: '{clase_gan}'") 
        else:
            # print(f"      🔹 Base semántica GAN: '{clase_gan}'")
            pass
        
        idx = self.le.transform([clase_gan])[0]
        z = np.random.normal(0, 1, (1, 128)).astype("float32")
        c = np.array([[idx]], dtype="int32")
        s = np.array([[0]], dtype="int32")
        
        img_gan = self.gan.predict([z, c, s], verbose=0)[0]
        img_gan = ((img_gan + 1.0) * 127.5).astype(np.uint8)
        base_pil = Image.fromarray(img_gan).resize((512, 512), Image.LANCZOS)

        prompt = f"A masterpiece digital painting of {tema}, poemario_style, intricate textures, cinematic lighting, ethereal, 8k"
        negative = "low quality, blurry, text, watermark, bad anatomy, deformed"

        final_img = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=base_pil,
            strength=0.75, 
            num_inference_steps=30,
            guidance_scale=8.0
        ).images[0]
        
        return final_img

# --- 5. EDITOR ---
def estampar(imagen, texto):
    draw = ImageDraw.Draw(imagen)
    w, h = imagen.size
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()

    try: bbox = draw.textbbox((0, 0), texto, font=font)
    except: bbox = draw.textsize(texto, font=font)
    
    text_w = bbox[2] - bbox[0] if len(bbox)==4 else bbox[0]
    text_h = bbox[3] - bbox[1] if len(bbox)==4 else bbox[1]

    x, y = (w - text_w) / 2, (h - text_h) / 2

    # Outline negro reforzado
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx != 0 or dy != 0:
                draw.text((x+dx, y+dy), texto, font=font, fill="black", align="center")
    
    draw.text((x, y), texto, font=font, fill="white", align="center")
    return imagen

# --- 6. MAIN ---
def main():
    print(f"--- 🚀 INICIANDO TEST EXTREMO (50 CONCEPTOS ÚNICOS) ---")
    
    poeta = GeneradorTexto()
    cache_trabajo = []
    
    # Obtenemos las reglas y las mezclamos para dar variedad
    tipos_poema = list(REGLAS_ESTRUCTURA.keys())
    random.shuffle(tipos_poema)
    
    # Aseguramos que las listas tengan el mismo tamaño (cortamos o repetimos si hace falta)
    # Pero aquí TEMAS_EXTREMOS son 50 y TIPOS son 50 aprox, así que usamos zip
    
    pares_trabajo = list(zip(TEMAS_EXTREMOS, tipos_poema))
    
    print(f"\n✍️ Escribiendo {len(pares_trabajo)} poemas extremos...")
    
    for idx, (tema, tipo) in enumerate(pares_trabajo):
        instruccion = REGLAS_ESTRUCTURA[tipo]
        print(f"   [{idx+1}/{len(pares_trabajo)}] {tipo} -> '{tema}'")
        
        poema = poeta.generar(tema, tipo, instruccion)
        cache_trabajo.append({"tema": tema, "tipo": tipo, "poema": poema})
    
    poeta.liberar()
    
    pincel = GeneradorImagen()
    print(f"\n🎨 Generando galería surrealista...")
    
    for idx, item in enumerate(cache_trabajo):
        tema = item["tema"]
        tipo = item["tipo"]
        poema = item["poema"]
        
        print(f"   [{idx+1}/{len(cache_trabajo)}] Pintando '{tema}'")
        
        img = pincel.crear_hibrido(tema)
        img_final = estampar(img, poema)
        
        clean_tipo = tipo.replace(" ", "_").lower()
        clean_tema = tema.replace(" ", "_")[:15].lower()
        filename = f"extreme_{idx+1:02d}_{clean_tipo}_{clean_tema}.png"
        
        img_final.save(os.path.join(OUTPUT_DIR, filename))
        print(f"      💾 {filename}")

    print(f"\n✅ ¡GALERÍA EXTREMA COMPLETADA! Carpeta: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()