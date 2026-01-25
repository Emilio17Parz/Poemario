from diffusers import StableDiffusionPipeline
import torch

# 1. Cargar SD 1.5
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
).to("cuda")

# 2. CARGAR TU LORA (Tu archivo final de Kohya)
# Reemplaza esta ruta con la ubicación de tu .safetensors
pipe.load_lora_weights(r"C:\Users\calza\Poemario\lora_outputs\poemario_v1.safetensors")

# 3. Prompt con tu Trigger Word
# IMPORTANTE: Pon siempre 'poemario_style' para activar tu estilo
prompt = "a professional digital painting of a soul, poemario_style, cinematic, hyperrealistic"
negative = "low quality, blurry, text, watermark, bad anatomy"

print("🎨 Generando con Stable Diffusion + Tu Estilo...")
image = pipe(prompt, negative_prompt=negative, num_inference_steps=30).images[0]
image.save("resultado_hibrido.png")
image.show()