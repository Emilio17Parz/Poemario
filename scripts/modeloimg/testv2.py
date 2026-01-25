import os
import torch
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler

# --- CONFIGURACIÓN DE RUTAS ---
GAN_PATH = r"C:\Users\calza\Poemario\saved_models_final\gen_final.keras"
ENCODER_PATH = r"C:\Users\calza\Poemario\saved_models_final\encoder_con.pkl"
LORA_PATH = r"C:\Users\calza\Poemario\saved_models_final\stable\poemario.safetensors"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = r"C:\Users\calza\Poemario\resultados_finales_totales"

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# --- CARGA DE MODELOS ---
print("⏳ Cargando motores...")
le = joblib.load(ENCODER_PATH)
gan = tf.keras.models.load_model(GAN_PATH, compile=False)

# Cargamos la versión de Texto a Imagen por defecto
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload() # Vital para tu RTX 4070
pipe.load_lora_weights(LORA_PATH)

# Creamos la versión de Imagen a Imagen compartiendo el mismo "cerebro" para ahorrar RAM
pipe_img2img = StableDiffusionImg2ImgPipeline(**pipe.components)

# --- LISTA DE PRUEBA AMPLIADA ---
conceptos = [
    "Belleza", "Egloga_Pastoril", "Expansion_Etereo", 
    "Desamor_Tristeza", "Gloria", "Soledad Cosmica", 
    "Atardecer de fuego", "Expansion_Texturas"
]

def generar_todo():
    clases_gan = list(le.classes_)
    
    for concepto in conceptos:
        prompt = f"A masterpiece digital painting of {concepto.replace('_', ' ')}, poemario_style, warm golden lighting, cinematic, 8k"
        negative = "low quality, blurry, text, watermark, bad anatomy"
        
        # BUSCAR SI LA GAN TIENE EL CONCEPTO
        encontrado = next((c for c in clases_gan if c.lower() == concepto.strip().lower()), None)
        
        if encontrado:
            print(f"\n🎨 [HÍBRIDO] Usando GAN + LoRA para: '{encontrado}'")
            idx = le.transform([encontrado])[0]
            z = np.random.normal(0, 1, (1, 128)).astype("float32")
            c = np.array([[idx]], dtype="int32")
            s = np.array([[0]], dtype="int32")
            
            img_gan = gan.predict([z, c, s], verbose=0)[0]
            img_gan = ((img_gan + 1.0) * 127.5).astype(np.uint8)
            base_pil = Image.fromarray(img_gan).resize((512, 512), Image.LANCZOS)
            
            final_img = pipe_img2img(prompt=prompt, negative_prompt=negative, image=base_pil, strength=0.6).images[0]
            tipo = "hibrido"
        else:
            print(f"\n✨ [PURO] Usando puro LoRA para: '{concepto}'")
            final_img = pipe(prompt=prompt, negative_prompt=negative, num_inference_steps=30).images[0]
            tipo = "puro"

        final_img.save(os.path.join(OUTPUT_DIR, f"{tipo}_{concepto.lower().replace(' ', '_')}.png"))

if __name__ == "__main__":
    generar_todo()
    print(f"\n✅ Proceso terminado en: {OUTPUT_DIR}")