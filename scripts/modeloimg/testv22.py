import os
import torch
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

# --- 1. CONFIGURACIÓN DE MEMORIA (RTX 4070) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e: print(e)

# --- 2. RUTAS ---
GAN_PATH = r"C:\Users\calza\Poemario\saved_models_final\gen_final.keras"
ENCODER_PATH = r"C:\Users\calza\Poemario\saved_models_final\encoder_con.pkl"
LORA_PATH = r"C:\Users\calza\Poemario\saved_models_final\stable\poemario.safetensors"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = r"C:\Users\calza\Poemario\resultados_hibridos_final"

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# --- 3. CARGA DE MODELOS ---
print("⏳ Cargando GAN (Keras) y Diccionario...")
gan = tf.keras.models.load_model(GAN_PATH, compile=False)
le = joblib.load(ENCODER_PATH)

print("🚀 Cargando Stable Diffusion + Tu LoRA...")
# Usamos Img2ImgPipeline para procesar la base de la GAN
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload() # Vital para tus 8GB de VRAM
pipe.load_lora_weights(LORA_PATH)

# --- 4. LISTA DE PRUEBA ---
conceptos = [
    "Amor", 
    "Aventura_Heroismo", 
    "Expansion_Cyberpunk", 
    "Haiku_Elementos", 
    "Zafiros En La Niebla",
    "Arquitectura eterea"
]

def procesar_galeria():
    clases = list(le.classes_)
    
    for concepto in conceptos:
        concepto_limpio = concepto.strip()
        # Búsqueda que ignora mayúsculas/minúsculas
        encontrado = next((c for c in clases if c.lower() == concepto_limpio.lower()), None)
        
        if not encontrado:
            print(f"⚠️ Saltando '{concepto}': no está en el dataset de la GAN.")
            continue

        print(f"\n🎨 Creando híbrido para: '{encontrado}'")
        
        # A. GENERAR BASE CON GAN
        idx = le.transform([encontrado])[0]
        z = np.random.normal(0, 1, (1, 128)).astype("float32")
        c = np.array([[idx]], dtype="int32")
        s = np.array([[0]], dtype="int32")
        
        img_gan = gan.predict([z, c, s], verbose=0)[0]
        img_gan = ((img_gan + 1.0) * 127.5).astype(np.uint8)
        base_pil = Image.fromarray(img_gan).resize((512, 512), Image.LANCZOS)

        # B. REFINAR CON STABLE DIFFUSION + LORA
        prompt = f"A masterpiece digital painting of {encontrado.replace('_', ' ')}, poemario_style, intricate textures, cinematic lighting, ethereal, 8k"
        negative = "low quality, blurry, text, watermark, bad anatomy, deformed"

        # 'strength' controla qué tanto se parece a la mancha de la GAN (0.6 es ideal)
        final_img = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=base_pil,
            strength=0.65, 
            num_inference_steps=30,
            guidance_scale=8.0
        ).images[0]

        # Guardar resultado hibridado
        final_img.save(os.path.join(OUTPUT_DIR, f"hibrido_{encontrado.lower()}.png"))
        print(f"✅ Guardado: hibrido_{encontrado.lower()}.png")

if __name__ == "__main__":
    procesar_galeria()
    print(f"\n✨ ¡Galería terminada! Revisa: {OUTPUT_DIR}")