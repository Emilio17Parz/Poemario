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
OUTPUT_DIR = os.path.join(BASE_DIR, "galeria_smart_text_v3")

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

# --- 2. LISTA DE TEMAS ---
TEMAS_EXTREMOS = [
    "relojes derritiendose en el desierto", "lluvia de paraguas negros", 
    "un ojo gigante en el cielo", "escaleras que suben a la nada", 
    "el ajedrez infinito", "peces nadando en el aire",
    "ciborg meditando en un templo", "autopista de datos neon", 
    "la ultima bateria de la tierra", "ciudad flotante entre nubes toxicas", 
    "android soñando con ovejas electricas", "mercado nocturno en marte",
    "catedral hecha de huesos", "el susurro que sale del espejo", 
    "bosque de arboles con manos", "la sombra que tiene vida propia", 
    "un payaso triste en un hospital abandonado", "el rey de las ratas",
    "volcan erupcionando hielo azul", "bosque de cristales gigantes", 
    "bioluminiscencia en la playa negra", "aurora boreal sobre piramides", 
    "tornado de fuego", "jardin de plantas carnivoras gigantes",
    "la geometria sagrada del alma", "el eco del vacio", 
    "la paradoja de schrodinger", "el sonido del silencio", 
    "nostalgia de un tiempo que no existio", "el caos cuantico",
    "una maquina de escribir fantasma", "un barco en una botella rota", 
    "mascara de oro llorando sangre", "una biblioteca laberintica", 
    "trebol de cinco hojas de metal",
    "batalla de angeles caidos", "el fin del universo", 
    "nacimiento de una estrella", "ruinas de una civilizacion acuatica", 
    "el trono del rey olvidado",
    "la soledad de un dios", "la furia de la tormenta", 
    "esperanza en una caja de pandora", "el primer beso del apocalipsis",
    "un gato hecho de humo", "mariposas mecanicas oxidadas", 
    "el vals de los fantasmas", "arquitectura imposible"
]
TEMAS_EXTREMOS = TEMAS_EXTREMOS[:50] 

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

# Límites de líneas para tijera (reducidos para garantizar espacio en imagen)
LIMITES_LINEAS = {
    "Pareado": 2, "Haiku": 3, "Terceto": 3, "Tanka": 5, 
    "Cuarteta": 4, "Redondilla": 4, "Cuarteto": 4, "Serventesio": 4,
    "Copla": 4, "Seguidilla": 4, "Estrofa alcaica": 4, "Estrofa safica": 4,
    "Limerick": 5, "Lira": 5, "Quintilla": 5,
    "Triolet": 8, "Decima espinela": 10, "Soneto": 14, 
    "Rondeau": 15, "Villanelle": 19,
    # Límites visuales ESTRICTOS para los largos
    "Sestina": 10, "Pantoum": 10, "Romance": 10, "Silva": 10, 
    "Verso libre": 8, "Poema en prosa": 6, "Zejel": 8, "Moaxaja": 8, 
    "Madrigal": 8, "Ghazal": 8, "Gacela": 8, "Balada": 10, "Oda": 10, 
    "Elegia": 10, "Himno": 10
}

PALETAS_COLOR = {
    "fuego": "vibrant red, orange and gold magma colors, warm lighting",
    "volcan": "intense reds, charred blacks, molten lava glow",
    "guerra": "gritty earth tones, rusted metal, fiery orange explosions, dark atmosphere",
    "ira": "aggressive crimson and black palette, high contrast",
    "oceano": "deep blues, aqua, turquoise, bioluminescent greens, cool tones",
    "agua": "aquatic palette, clear blues and teals, reflective surfaces",
    "hielo": "icy whites, glacial blues, frosted silver, cold atmosphere",
    "frio": "frozen tones, desaturated blues and grays",
    "naturaleza": "lush greens, earthy browns, floral colors, organic palette",
    "bosque": "deep forest greens, mossy textures, dappled sunlight",
    "jardin": "vibrant floral colors, pinks, purples, yellows and greens",
    "cyberpunk": "neon pinks, electric blues, purples, dark metallic tones, futuristic glow",
    "neon": "glowing neon colors against dark backgrounds, high saturation",
    "android": "chrome, circuitry copper, LED blue and red lights",
    "sueños": "soft pastel colors, dreamy haze, iridescent pinks and blues",
    "surrealismo": "bizarre and contrasting color combinations, dreamlike quality",
    "etereo": "soft whites, gold, pearlescent colors, heavenly glow",
    "oscuridad": "deep blacks, charcoal, moody shadows with subtle highlights",
    "vacio": "monochrome, minimalist, high contrast black and white with one accent color",
    "muerte": "somber tones, deep purples, grays, faded colors",
    "oro": "rich gold, metallic sheen, luxurious warm tones",
    "sangre": "deep crimson and ruby red palette"
}
PALETAS_DEFAULT = [
    "rich and varied polychromatic palette",
    "high contrast vibrant colors",
    "muted and moody earth tones",
    "iridescent and pearlescent colors"
]

# --- 3. CLASE POETA ---
class GeneradorTexto:
    def __init__(self):
        print("📖 [1/2] Cargando Poeta...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(POEM_MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(POEM_MODEL_DIR).to(self.device)

    def limpiar_y_cortar(self, texto, tipo_poema):
        # 1. Quitar líneas vacías
        lineas = [l.strip() for l in texto.split('\n') if l.strip()]
        
        # 2. Cortar por límite de líneas (Tijera Vertical)
        limite = LIMITES_LINEAS.get(tipo_poema, 8)
        if len(lineas) > limite:
            lineas = lineas[:limite]
        
        return "\n".join(lineas)

    def generar(self, tema, tipo_poema, instruccion):
        orden_extra = f"El poema debe incluir explícitamente la frase '{tema}'."
        prompt = f"TIPO: {tipo_poema}\nINSTRUCCION: {instruccion} {orden_extra}\nTEMA: {tema}\n### POEMA:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        limite_lineas = LIMITES_LINEAS.get(tipo_poema, 14)
        max_len = min(380, limite_lineas * 25 + len(inputs["input_ids"][0]))

        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_len, 
                do_sample=True, temperature=0.85, top_p=0.92, repetition_penalty=1.2,
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

# --- 4. CLASE PINCEL ---
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

    def obtener_paleta_color(self, tema):
        tema_lower = tema.lower()
        paleta_elegida = ""
        for keyword, palette in PALETAS_COLOR.items():
            if keyword in tema_lower:
                paleta_elegida = palette
                break
        if not paleta_elegida:
            paleta_elegida = random.choice(PALETAS_DEFAULT)
        return paleta_elegida

    def crear_hibrido(self, tema):
        tema_limpio = tema.strip().lower()
        clase_gan = None
        for palabra in tema_limpio.split():
             match = next((c for c in self.clases_gan if palabra in c.lower()), None)
             if match:
                 clase_gan = match
                 break
        if not clase_gan: clase_gan = random.choice(self.clases_gan)
        
        idx = self.le.transform([clase_gan])[0]
        z = np.random.normal(0, 1, (1, 128)).astype("float32")
        c = np.array([[idx]], dtype="int32")
        s = np.array([[0]], dtype="int32")
        
        img_gan = self.gan.predict([z, c, s], verbose=0)[0]
        img_gan = ((img_gan + 1.0) * 127.5).astype(np.uint8)
        base_pil = Image.fromarray(img_gan).resize((512, 512), Image.LANCZOS)

        color_prompt = self.obtener_paleta_color(tema)
        prompt = f"A masterpiece digital painting of {tema}, {color_prompt}, poemario_style, intricate textures, cinematic lighting, ethereal, 8k"
        negative = "low quality, blurry, text, watermark, bad anatomy, deformed, monochrome, desaturated, muted colors, sepia tone, boring colors"

        final_img = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=base_pil,
            strength=0.75, 
            num_inference_steps=30,
            guidance_scale=8.5
        ).images[0]
        
        return final_img

# --- 5. EDITOR INTELIGENTE (WRAP & RESIZE) ---
def estampar_inteligente(imagen, texto):
    draw = ImageDraw.Draw(imagen)
    W, H = imagen.size
    
    # Márgenes de seguridad (para que no toque los bordes)
    margin = 40
    max_width = W - (margin * 2)
    max_height = H - (margin * 2)

    # Empezamos con fuente tamaño 26, bajamos hasta que quepa
    font_size = 26
    min_font_size = 12
    
    try: font_path = "arial.ttf"
    except: font_path = None # Usará default si falla

    final_lines = []
    final_font = None

    # Bucle para encontrar el tamaño de fuente correcto
    while font_size >= min_font_size:
        try:
            if font_path: font = ImageFont.truetype(font_path, font_size)
            else: font = ImageFont.load_default() # Default no escala bien, pero es fallback
        except:
             font = ImageFont.load_default()

        # LOGICA DE WRAPPING (Cortar líneas largas)
        lines = []
        paragraphs = texto.split('\n')
        
        for p in paragraphs:
            words = p.split()
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                
                # Medir ancho de la línea de prueba
                try: bbox = draw.textbbox((0, 0), test_line, font=font)
                except: bbox = draw.textsize(test_line, font=font)
                
                w_line = bbox[2] - bbox[0] if len(bbox)==4 else bbox[0]
                
                if w_line <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line: lines.append(current_line)
        
        # Calcular altura total de todas las líneas envueltas
        total_text_height = 0
        line_heights = []
        for line in lines:
            try: bbox = draw.textbbox((0, 0), line, font=font)
            except: bbox = draw.textsize(line, font=font)
            h_line = bbox[3] - bbox[1] if len(bbox)==4 else bbox[1]
            # Añadimos un poco de espaciado entre líneas
            h_line_spaced = h_line + 5 
            line_heights.append(h_line_spaced)
            total_text_height += h_line_spaced

        # VERIFICAR SI CABE EN ALTO
        if total_text_height <= max_height:
            # ¡Cabe! Guardamos y salimos del bucle
            final_lines = lines
            final_font = font
            break
        else:
            # No cabe, reducimos fuente y reintentamos
            font_size -= 2
    
    # Si salimos del bucle y no tenemos fuente (texto gigante), usamos la mínima
    if final_font is None:
        try: final_font = ImageFont.truetype("arial.ttf", min_font_size)
        except: final_font = ImageFont.load_default()
        final_lines = texto.split('\n') # Fallback simple

    # DIBUJAR (Ahora sí, centrado)
    # Recalculamos altura final
    total_h = sum([draw.textbbox((0,0), l, font=final_font)[3] - draw.textbbox((0,0), l, font=final_font)[1] + 5 for l in final_lines])
    
    current_y = (H - total_h) / 2
    
    for line in final_lines:
        try: bbox = draw.textbbox((0, 0), line, font=final_font)
        except: bbox = draw.textsize(line, font=final_font)
        w_line = bbox[2] - bbox[0] if len(bbox)==4 else bbox[0]
        h_line = bbox[3] - bbox[1] if len(bbox)==4 else bbox[1]
        
        current_x = (W - w_line) / 2
        
        # Borde Negro Reforzado
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx!=0 or dy!=0:
                    draw.text((current_x+dx, current_y+dy), line, font=final_font, fill="black")
        
        # Texto Blanco
        draw.text((current_x, current_y), line, font=final_font, fill="white")
        
        current_y += h_line + 5 # Salto de línea

    return imagen

# --- 6. MAIN ---
def main():
    print(f"--- 🚀 INICIANDO TEST FINAL V3 (SMART TEXT) ---")
    
    poeta = GeneradorTexto()
    cache_trabajo = []
    
    tipos_poema = list(REGLAS_ESTRUCTURA.keys())
    random.shuffle(tipos_poema)
    pares_trabajo = list(zip(TEMAS_EXTREMOS, tipos_poema))
    
    print(f"\n✍️ Escribiendo {len(pares_trabajo)} poemas...")
    
    for idx, (tema, tipo) in enumerate(pares_trabajo):
        instruccion = REGLAS_ESTRUCTURA[tipo]
        print(f"   [{idx+1}/{len(pares_trabajo)}] {tipo} -> '{tema}'")
        poema = poeta.generar(tema, tipo, instruccion)
        cache_trabajo.append({"tema": tema, "tipo": tipo, "poema": poema})
    
    poeta.liberar()
    
    pincel = GeneradorImagen()
    print(f"\n🎨 Generando galería con texto inteligente...")
    
    for idx, item in enumerate(cache_trabajo):
        tema = item["tema"]
        tipo = item["tipo"]
        poema = item["poema"]
        
        print(f"   [{idx+1}/{len(cache_trabajo)}] Pintando '{tema}'")
        
        img = pincel.crear_hibrido(tema)
        # USAMOS LA NUEVA FUNCIÓN INTELIGENTE
        img_final = estampar_inteligente(img, poema)
        
        clean_tipo = tipo.replace(" ", "_").lower()
        clean_tema = tema.replace(" ", "_")[:15].lower()
        filename = f"smart_{idx+1:02d}_{clean_tipo}_{clean_tema}.png"
        
        img_final.save(os.path.join(OUTPUT_DIR, filename))
        print(f"      💾 {filename}")

    print(f"\n✅ ¡GALERÍA SMART COMPLETADA! Carpeta: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()