import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Ruta donde se guardó el modelo entrenado
MODEL_PATH = "./modelo_poemas_entrenado"

def generar_poema(tipo, keyword):
    try:
        # Cargar modelo y tokenizer entrenados
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        
        # Detectar si hay GPU disponible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        # Construir el prompt con el MISMO formato que el entrenamiento
        prompt = f"<|startoftext|>TIPO: {tipo} | KEYWORD: {keyword} | POEMA:"
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Generar
        print(f"Generando poema de tipo '{tipo}' sobre '{keyword}'...")
        
        sample_outputs = model.generate(
            input_ids,
            do_sample=True, 
            max_length=300,         # Longitud máxima generada
            top_k=50,               # Muestreo top-k
            top_p=0.95,             # Nucleus sampling (creatividad vs coherencia)
            temperature=0.8,        # Creatividad (más alto = más loco)
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id # Detenerse cuando el modelo quiera terminar
        )

        for i, sample_output in enumerate(sample_outputs):
            decoded_text = tokenizer.decode(sample_output, skip_special_tokens=True)
            
            # Limpiamos el prompt del resultado final para mostrar solo el poema
            # El texto decodificado se verá como: "TIPO: X | KEYWORD: Y | POEMA: El poema..."
            partes = decoded_text.split("POEMA:")
            if len(partes) > 1:
                poema_limpio = partes[1].strip()
                print("\n" + "="*30)
                print(f"✨ POEMA GENERADO:\n\n{poema_limpio}")
                print("="*30 + "\n")
            else:
                print(decoded_text)

    except Exception as e:
        print(f"Error al cargar el modelo o generar: {e}")
        print("Asegúrate de haber ejecutado primero el script de entrenamiento.")

if __name__ == "__main__":
    # --- PRUEBA MANUAL ---
    print("--- GENERADOR DE POEMAS IA ---")
    tipo_input = input("Ingresa el tipo de poema (ej. Soneto, Haiku, Libre): ")
    keyword_input = input("Ingresa la palabra clave o frase (ej. Luna, El paso del tiempo): ")
    
    generar_poema(tipo_input, keyword_input)