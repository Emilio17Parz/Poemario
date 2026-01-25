import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# --- CONFIGURACIÓN ---
# Usamos el modelo base que descargó Kohya
MODEL_ID = "runwayml/stable-diffusion-v1-5"
# La ruta del archivo que acabas de terminar de entrenar
LORA_PATH = r"C:\Users\calza\Poemario\saved_models_final\stable\poemario.safetensors"

print("🚀 Cargando el cerebro de Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

print("🧠 Inyectando tu estilo 'Poemario'...")
pipe.load_lora_weights(LORA_PATH)

# --- EL PROMPT MÁGICO ---
# Usa 'poemario_style' + un concepto de tus carpetas (ej: Zafiros En La Niebla)
prompt = "a mystical landscape of zafiros en la niebla, poemario_style, professional oil painting, cinematic, 8k"
negative = "low quality, blurry, text, watermark, bad anatomy, person, face"

print("🎨 Generando obra maestra...")
# Generamos 4 imágenes para comparar
for i in range(4):
    image = pipe(prompt, negative_prompt=negative, num_inference_steps=30).images[0]
    image.save(f"poema_visual_{i}.png")

print("✅ ¡Listo! Revisa las imágenes 'poema_visual_X.png' en tu carpeta.")