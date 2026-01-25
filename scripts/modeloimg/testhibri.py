import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os

# --- CONFIGURACIÓN ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = r"C:\Users\calza\Poemario\saved_models_final\stable\poemario.safetensors"
OUTPUT_DIR = r"C:\Users\calza\Poemario\resultados_test"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Lista de conceptos para probar el modelo
conceptos = [
    "Amor", 
    "Aventura_Heroismo", 
    "Expansion_Cyberpunk", 
    "Haiku_Elementos", 
    "Zafiros En La Niebla",
    "Un gato cosmico", # Algo fuera del dataset para ver la mezcla
    "Arquitectura eterea"
]

print("🚀 Cargando cerebro de Stable Diffusion...")
# Usamos dtype en lugar del deprecado torch_dtype
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Optimizaciones críticas para 8GB de VRAM
pipe.enable_model_cpu_offload() 

print("🧠 Inyectando estilo Poemario (LoRA)...")
pipe.load_lora_weights(LORA_PATH)

print(f"🎨 Iniciando galería de {len(conceptos)} conceptos...")

for concepto in conceptos:
    # Construcción del prompt usando el trigger entrenado
    prompt = f"A professional digital painting of {concepto.replace('_', ' ')}, poemario_style, artistic composition, cinematic lighting, 8k"
    negative = "low quality, blurry, text, watermark, deformed, ugly, bad anatomy"
    
    print(f"  -> Generando: {concepto}...")
    
    # Generación de la imagen
    image = pipe(
        prompt, 
        negative_prompt=negative, 
        num_inference_steps=30, 
        guidance_scale=8.0 # Sube este valor para que sea más fiel al prompt
    ).images[0]
    
    # Guardado con nombre único
    filename = f"test_{concepto.lower().replace(' ', '_')}.png"
    image.save(os.path.join(OUTPUT_DIR, filename))

print(f"\n✅ ¡Pruebas terminadas! Revisa la carpeta: {OUTPUT_DIR}")