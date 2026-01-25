import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# 1. Configuración de Rutas
model_id = "runwayml/stable-diffusion-v1-5" # Modelo base
lora_path = r"C:\Users\calza\Poemario\output\mi_estilo_poemario.safetensors"

# 2. Cargar el Pipeline
print("🚀 Cargando Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# 3. CARGAR TU LORA (Tu conocimiento entrenado)
# Esto añade tu estilo al modelo gigante
print("🧠 Aplicando tu estilo LoRA...")
pipe.load_lora_weights(lora_path)

# 4. Generar con tu Trigger Word
# Importante: Incluir 'poemario_style' para activar tu entrenamiento
prompt = "a digital painting of a cosmic rose, poemario_style, high quality, masterpiece"
negative_prompt = "low quality, blurry, distorted, text, watermark"

print("🎨 Generando imagen...")
image = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    num_inference_steps=30, 
    guidance_scale=7.5
).images[0]

# 5. Guardar
image.save("resultado_poemario_sd.png")
image.show()