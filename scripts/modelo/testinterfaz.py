import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import shutil
import torch
import numpy as np
import tensorflow as tf
import joblib
import random
import re
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

# *** CAMBIA ESTO POR TU IMAGEN DE FONDO ***
RUTA_FONDO_GUI = r"C:\Users\calza\Poemario\scripts\modelo\gato.png" 

BASE_DIR = r"C:\Users\calza\Poemario"
OUTPUT_DIR = os.path.join(BASE_DIR, "galeria_final_gui")

GAN_PATH = os.path.join(BASE_DIR, r"saved_models_final\gen_final.keras")
ENCODER_PATH = os.path.join(BASE_DIR, r"saved_models_final\encoder_con.pkl")
LORA_PATH = os.path.join(BASE_DIR, r"saved_models_final\stable\poemario.safetensors")
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
POEM_MODEL_DIR = os.path.join(BASE_DIR, "modelo_poemas_neo_balanceado_v3")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. REGLAS Y DICCIONARIOS
# ==========================================

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

LIMITES_LINEAS = {
    "Pareado": 2, "Haiku": 3, "Terceto": 3, "Tanka": 5, 
    "Cuarteta": 4, "Redondilla": 4, "Cuarteto": 4, "Serventesio": 4,
    "Copla": 4, "Seguidilla": 4, "Estrofa alcaica": 4, "Estrofa safica": 4,
    "Limerick": 5, "Lira": 5, "Quintilla": 5,
    "Triolet": 8, "Decima espinela": 10, "Soneto": 14, 
    "Rondeau": 15, "Villanelle": 19,
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

# ==========================================
# 3. CLASES LÓGICAS (Con Lógica de Puntuación)
# ==========================================

class GeneradorTexto:
    def __init__(self):
        print("📖 Cargando Poeta...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def cargar(self):
        self.tokenizer = AutoTokenizer.from_pretrained(POEM_MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(POEM_MODEL_DIR).to(self.device)

    def limpiar_y_cortar(self, texto, tipo_poema):
        lineas = [l.strip() for l in texto.split('\n') if l.strip()]
        limite = LIMITES_LINEAS.get(tipo_poema, 8)
        if len(lineas) > limite:
            lineas = lineas[:limite]
        
        texto_unido = "\n".join(lineas)
        puntuacion_match = list(re.finditer(r'[.,]', texto_unido))
        
        if puntuacion_match:
            ultimo_signo = puntuacion_match[-1]
            corte_index = ultimo_signo.end()
            texto_final = texto_unido[:corte_index]
            if texto_final.endswith(','):
                texto_final = texto_final[:-1] + '.'
            return texto_final
        else:
            return texto_unido + "."

    def generar(self, tema, tipo_poema, instruccion):
        if not self.model: self.cargar()
        
        palabras_clave = [p.lower() for p in tema.split() if len(p) > 3]
        if not palabras_clave: palabras_clave = [tema.lower()]

        best_candidate = ""
        intentos_max = 3

        for intento in range(intentos_max):
            orden_extra = f"IMPORTANTE: Es OBLIGATORIO incluir explícitamente la palabra o frase '{tema}' dentro del poema."
            prompt = f"TIPO: {tipo_poema}\nINSTRUCCION: {instruccion} {orden_extra}\nTEMA: {tema}\n### POEMA:\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            limite_lineas = LIMITES_LINEAS.get(tipo_poema, 14)
            max_len = min(400, limite_lineas * 25 + len(inputs["input_ids"][0]))

            temp_actual = 0.85 + (intento * 0.02) 

            with torch.no_grad():
                out = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_len, 
                    do_sample=True, 
                    temperature=temp_actual, 
                    top_p=0.92, 
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            raw = self.tokenizer.decode(out[0], skip_special_tokens=True)
            if "### POEMA:" in raw: texto = raw.split("### POEMA:")[1].strip()
            else: texto = raw.replace(prompt, "").strip()
            
            texto_limpio = self.limpiar_y_cortar(texto, tipo_poema)
            
            encontrado = False
            texto_lower = texto_limpio.lower()
            if tema.lower() in texto_lower: encontrado = True
            else:
                for p in palabras_clave:
                    if p in texto_lower:
                        encontrado = True; break
            
            if encontrado: return texto_limpio
            else: best_candidate = texto_limpio

        return best_candidate

class GeneradorImagen:
    def __init__(self):
        self.gan = None
        self.pipe = None
        self.le = None
        self.clases_gan = []

    def cargar(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e: print(e)

        self.gan = tf.keras.models.load_model(GAN_PATH, compile=False)
        self.le = joblib.load(ENCODER_PATH)
        self.clases_gan = list(self.le.classes_)
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.load_lora_weights(LORA_PATH)

    def obtener_paleta_color(self, tema):
        tema_lower = tema.lower()
        for k, v in PALETAS_COLOR.items():
            if k in tema_lower: return v
        return random.choice(PALETAS_DEFAULT)

    def crear_hibrido(self, tema):
        if not self.gan: self.cargar()
        tema_limpio = tema.strip().lower()
        palabras_tema = tema_limpio.split()
        
        clase_gan = None
        for c in self.clases_gan:
            for p in palabras_tema:
                if p in c.lower():
                    clase_gan = c; break
            if clase_gan: break
        
        if not clase_gan: clase_gan = random.choice(self.clases_gan)
        
        idx = self.le.transform([clase_gan])[0]
        z = np.random.normal(0, 1, (1, 128)).astype("float32")
        c_gan = np.array([[idx]], dtype="int32")
        s = np.array([[0]], dtype="int32")
        
        img_gan = self.gan.predict([z, c_gan, s], verbose=0)[0]
        img_gan = ((img_gan + 1.0) * 127.5).astype(np.uint8)
        base_pil = Image.fromarray(img_gan).resize((512, 512), Image.LANCZOS)

        color_prompt = self.obtener_paleta_color(tema)
        prompt = f"A masterpiece digital painting of {tema}, {color_prompt}, poemario_style, intricate textures, cinematic lighting, ethereal, 8k"
        negative = "low quality, blurry, text, watermark, bad anatomy, deformed"

        final_img = self.pipe(
            prompt=prompt, negative_prompt=negative, image=base_pil,
            strength=0.75, num_inference_steps=30, guidance_scale=8.5
        ).images[0]
        
        return final_img

# ==========================================
# 4. FUNCIÓN VISUAL
# ==========================================
def estampar_final(imagen, texto):
    draw = ImageDraw.Draw(imagen)
    W, H = imagen.size
    
    margin_x = W * 0.1
    margin_y = H * 0.1
    max_width_box = W - (2 * margin_x)
    max_height_box = H - (2 * margin_y)
    
    font_names = ["ariali.ttf", "calibrii.ttf", "arial.ttf"] 
    font_path = None
    for fn in font_names:
        try:
            ImageFont.truetype(fn, 20)
            font_path = fn
            break
        except: continue

    original_lines = texto.split('\n')
    font_size = 60
    min_font_size = 14
    final_font = None
    final_wrapped_lines = []

    while font_size >= min_font_size:
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except: font = ImageFont.load_default()

        current_wrapped_lines = []
        total_h_sim = 0
        line_spacing = font_size * 0.2
        fits_width = True
        
        for line in original_lines:
            words = line.split()
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                bbox = draw.textbbox((0, 0), test_line, font=font)
                w_line = bbox[2] - bbox[0]
                if w_line <= max_width_box: current_line = test_line
                else:
                    bbox_word = draw.textbbox((0, 0), word, font=font)
                    if (bbox_word[2]-bbox_word[0]) > max_width_box: fits_width = False; break
                    current_wrapped_lines.append(current_line)
                    current_line = word
            if not fits_width: break
            if current_line: current_wrapped_lines.append(current_line)
        
        if not fits_width:
            font_size -= 2; continue

        for line in current_wrapped_lines:
            bbox = draw.textbbox((0,0), line, font=font)
            h = bbox[3] - bbox[1]
            total_h_sim += h + line_spacing
        
        if total_h_sim <= max_height_box:
            final_font = font
            final_wrapped_lines = current_wrapped_lines
            break
        else: font_size -= 2

    if final_font is None:
        final_font = ImageFont.load_default()
        final_wrapped_lines = original_lines

    total_h = 0
    final_spacing = final_font.size * 0.2 if hasattr(final_font, 'size') else 5
    line_heights = []
    
    for line in final_wrapped_lines:
         bbox = draw.textbbox((0,0), line, font=final_font)
         h = bbox[3] - bbox[1]
         line_heights.append(h)
         total_h += h + final_spacing
    total_h -= final_spacing

    current_y = (H - total_h) / 2

    for i, line in enumerate(final_wrapped_lines):
        bbox = draw.textbbox((0, 0), line, font=final_font)
        w_line = bbox[2] - bbox[0]
        current_x = (W - w_line) / 2

        outline = max(1, int(final_font.size / 20)) if hasattr(final_font, 'size') else 1
        for dx in range(-outline, outline+1):
            for dy in range(-outline, outline+1):
                if dx!=0 or dy!=0: draw.text((current_x+dx, current_y+dy), line, font=final_font, fill="black")
        
        draw.text((current_x, current_y), line, font=final_font, fill="white")
        current_y += line_heights[i] + final_spacing

    return imagen

# ==========================================
# 5. INTERFAZ GRÁFICA (V11 - TEXTO FLOTANTE)
# ==========================================

class PoemarioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Poemario AI - Generador Final V11")
        self.root.geometry("950x750")
        self.root.resizable(False, False)

        self.bg_image_tk = None
        self.canvas_bg = None
        
        self.poeta = GeneradorTexto()
        self.pincel = GeneradorImagen()
        self.modelos_cargados = False
        self.current_image_path = None 

        self.setup_ui()
        
        self.status_var.set("Cargando sistema... Espere.")
        self.btn_generar.config(state="disabled")
        threading.Thread(target=self.cargar_modelos_thread, daemon=True).start()

    def setup_background_on_canvas(self):
        """Carga la imagen y la pone en un Canvas para poder dibujar texto encima"""
        try:
            if os.path.exists(RUTA_FONDO_GUI):
                pil_img = Image.open(RUTA_FONDO_GUI).resize((950, 750), Image.LANCZOS)
                self.bg_image_tk = ImageTk.PhotoImage(pil_img)
                
                # Creamos el Canvas que cubrirá toda la ventana
                self.canvas_bg = tk.Canvas(self.root, width=950, height=750, highlightthickness=0)
                self.canvas_bg.pack(fill="both", expand=True)
                
                # Ponemos la imagen en el canvas
                self.canvas_bg.create_image(0, 0, image=self.bg_image_tk, anchor="nw")
            else:
                self.root.configure(bg="#2c3e50")
                self.canvas_bg = tk.Canvas(self.root, width=950, height=750, bg="#2c3e50", highlightthickness=0)
                self.canvas_bg.pack(fill="both", expand=True)
        except Exception as e:
             print(f"Error: {e}")
             self.root.configure(bg="#2c3e50")

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCombobox", font=("Arial", 11))

        # 1. Crear Canvas y Fondo
        self.setup_background_on_canvas()

        # 2. Textos Flotantes (TRANSPARENTES - Sin cuadro negro)
        # Usamos create_text en lugar de Label
        
        # [NUEVO] Título Principal
        self.canvas_bg.create_text(240, 80, text="POEMARIO", font=("Times New Roman", 40, "bold"), fill="#f1c40f")

        # [NUEVO] Etiquetas de Inputs
        self.canvas_bg.create_text(240, 160, text="Tema del poema:", font=("Arial", 12, "bold"), fill="#ecf0f1", anchor="center")
        self.canvas_bg.create_text(240, 250, text="Estructura / Tipo:", font=("Arial", 12, "bold"), fill="#ecf0f1", anchor="center")

        # 3. Widgets Interactivos (Estos sí van 'encima' del canvas)
        
        self.entry_tema = tk.Entry(self.root, font=("Arial", 14), width=30)
        # Usamos place relativo a la ventana, quedará encima del canvas
        self.entry_tema.place(x=75, y=180)

        opciones = sorted(list(REGLAS_ESTRUCTURA.keys()))
        self.combo_tipo = ttk.Combobox(self.root, values=opciones, state="readonly", font=("Arial", 11), width=28)
        self.combo_tipo.current(0)
        self.combo_tipo.place(x=75, y=270)

        self.btn_generar = tk.Button(self.root, text=" GENERAR OBRA ", 
                                     font=("Arial", 13, "bold"), bg="#f1c40f", fg="#000000",
                                     command=self.iniciar_generacion, cursor="hand2", width=30)
        self.btn_generar.place(x=75, y=330)

        self.btn_descargar = tk.Button(self.root, text=" DESCARGAR IMAGEN", 
                                     font=("Arial", 11, "bold"), bg="#3498db", fg="white",
                                     state="disabled", command=self.descargar_imagen, cursor="hand2", width=35)
        self.btn_descargar.place(x=75, y=400)

        # 4. Estado y Preview
        # El estado lo dejamos como label pequeño para que se lea si el fondo es claro
        self.status_var = tk.StringVar(value="Cargando...")
        self.lbl_status = tk.Label(self.root, textvariable=self.status_var, 
                                   font=("Arial", 10, "italic"), bg="black", fg="#bdc3c7")
        self.lbl_status.place(x=75, y=650)

        # Preview de Imagen
        self.lbl_img_preview = tk.Label(self.root, text="Tu obra aparecerá aquí.", 
                                        font=("Arial", 14), bg="#1a1a1a", fg="#7f8c8d")
        self.lbl_img_preview.place(x=450, y=100, width=460, height=460)

    def cargar_modelos_thread(self):
        try:
            self.poeta.cargar()
            self.status_var.set("Cargando Pincel...")
            self.pincel.cargar()
            self.modelos_cargados = True
            self.root.after(0, lambda: self.status_var.set("¡Listo!"))
            self.root.after(0, lambda: self.btn_generar.config(state="normal"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

    def iniciar_generacion(self):
        tema = self.entry_tema.get().strip()
        tipo = self.combo_tipo.get()
        if not tema: return
        
        self.btn_generar.config(state="disabled", text="Trabajando...")
        self.btn_descargar.config(state="disabled")
        threading.Thread(target=self.proceso_generacion, args=(tema, tipo), daemon=True).start()

    def proceso_generacion(self, tema, tipo):
        try:
            self.root.after(0, lambda: self.status_var.set(f"Escribiendo sobre '{tema}'..."))
            instruccion = REGLAS_ESTRUCTURA[tipo]
            
            texto_poema = self.poeta.generar(tema, tipo, instruccion)
            
            self.root.after(0, lambda: self.status_var.set("Pintando imagen..."))
            img_base = self.pincel.crear_hibrido(tema)

            self.root.after(0, lambda: self.status_var.set("Estampando texto..."))
            img_final = estampar_final(img_base, texto_poema)
            
            clean_tipo = tipo.replace(" ", "_").lower()
            clean_tema = tema.replace(" ", "_").replace(".", "")[:15].lower()
            filename = f"GUI_{clean_tipo}_{clean_tema}.png"
            path_save = os.path.join(OUTPUT_DIR, filename)
            img_final.save(path_save)
            self.current_image_path = path_save

            self.root.after(0, lambda: self.mostrar_resultado(img_final))
        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", msg))
            self.root.after(0, self.restaurar_ui)

    def mostrar_resultado(self, pil_img):
        preview_img = pil_img.resize((460, 460), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(preview_img)
        self.current_image_ref = tk_img 
        self.lbl_img_preview.config(image=tk_img, text="", bg="black")
        self.status_var.set("¡Completado!")
        self.btn_descargar.config(state="normal", bg="#2ecc71")
        self.restaurar_ui()

    def restaurar_ui(self):
        self.btn_generar.config(state="normal", text="✨ GENERAR OBRA ✨")

    def descargar_imagen(self):
        if not self.current_image_path: return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")], initialfile=os.path.basename(self.current_image_path))
        if file_path:
            shutil.copy2(self.current_image_path, file_path)
            messagebox.showinfo("Guardado", f"Imagen guardada en:\n{file_path}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = PoemarioApp(root)
        root.mainloop()
    except Exception as e: print(e)