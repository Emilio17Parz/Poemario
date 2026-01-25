import os
import torch
import numpy as np
import tensorflow as tf
import joblib
import random
from PIL import Image
# Se han eliminado ImageDraw e ImageFont porque ya no se usará texto
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

# --- 1. CONFIGURACIÓN ---
BASE_DIR = r"C:\Users\calza\Poemario"
# Cambiamos el nombre de la carpeta de salida
OUTPUT_DIR = os.path.join(BASE_DIR, "galeria_imagenes_limpias_v1")

# Rutas Modelos de Imagen
GAN_PATH = os.path.join(BASE_DIR, r"saved_models_final\gen_final.keras")
ENCODER_PATH = os.path.join(BASE_DIR, r"saved_models_final\encoder_con.pkl")
LORA_PATH = os.path.join(BASE_DIR, r"saved_models_final\stable\poemario.safetensors")
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Se ha eliminado POEM_MODEL_DIR porque ya no se necesita el modelo de texto

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
# Si quieres generar los 50, descomenta la siguiente línea.
# TEMAS_EXTREMOS = TEMAS_EXTREMOS[:50] 

# --- Se han eliminado REGLAS_ESTRUCTURA y LIMITES_LINEAS ---

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

# --- SE HA ELIMINADO LA CLASE GeneradorTexto (POETA) ---

# --- 3. CLASE PINCEL (MANTENIDA IGUAL) ---
class GeneradorImagen:
    def __init__(self):
        print("🎨 Cargando Pincel (GAN + SD)...")
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

# --- SE HA ELIMINADO LA FUNCIÓN estampar_inteligente ---

# --- 4. MAIN MODIFICADO (SOLO IMAGENES) ---
def main():
    print(f"--- 🚀 INICIANDO GENERACIÓN DE IMÁGENES LIMPIAS ---")
    
    pincel = GeneradorImagen()
    total_imagenes = len(TEMAS_EXTREMOS)
    print(f"\n🎨 Generando {total_imagenes} imágenes sin texto...")
    
    # Iteramos directamente sobre la lista de temas
    for idx, tema in enumerate(TEMAS_EXTREMOS):
        print(f"   [{idx+1}/{total_imagenes}] Pintando '{tema}'")
        
        # 1. Generar la imagen híbrida
        img_final = pincel.crear_hibrido(tema)
        
        # 2. Preparar nombre de archivo (más simple, sin tipo de poema)
        clean_tema = tema.replace(" ", "_")[:20].lower() # Limitamos longitud del nombre
        filename = f"img_{idx+1:02d}_{clean_tema}.png"
        
        # 3. Guardar directamente la imagen limpia
        save_path = os.path.join(OUTPUT_DIR, filename)
        img_final.save(save_path)
        print(f"      💾 Guardada: {filename}")

    print(f"\n✅ ¡GALERÍA DE IMÁGENES LIMPIAS COMPLETADA! Carpeta: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()