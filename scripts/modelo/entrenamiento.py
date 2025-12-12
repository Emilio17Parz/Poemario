import json
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
MODEL_NAME = "PlanTL-GOB-ES/gpt2-base-bne" # Un excelente modelo base en español
OUTPUT_DIR = "./modelo_poemas_entrenado"
DATA_PATH = "./dataset/**/*.json" # Ajusta según donde tengas tus jsons
MAX_LENGTH = 256 # Longitud máxima del poema (tokens)

# --- 1. CARGA DE DATOS ---
def cargar_datos_desde_json(path_pattern):
    textos = []
    # Busca recursivamente archivos json
    files = glob.glob(path_pattern, recursive=True)
    
    print(f"Encontrados {len(files)} archivos.")
    
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extraemos los campos según tu esquema
                poema_obj = data.get('poema', {})
                texto = poema_obj.get('texto', '')
                tipo = poema_obj.get('tipo', 'Desconocido')
                keyword = poema_obj.get('palabra_clave_ingresada', '')
                
                if texto and keyword:
                    # Formateamos el string para que el modelo aprenda la estructura:
                    # INPUT (Condición) -> OUTPUT (Poema)
                    # Usamos tokens especiales para separar las partes
                    formato_entrenamiento = f"<|startoftext|>TIPO: {tipo} | KEYWORD: {keyword} | POEMA: {texto}<|endoftext|>"
                    textos.append(formato_entrenamiento)
        except Exception as e:
            print(f"Error leyendo {file}: {e}")
            
    return textos

# --- 2. DATASET PERSONALIZADO ---
class PoemasDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        
        for txt in txt_list:
            # Tokenizamos el texto completo
            encodings_dict = tokenizer(
                txt, 
                truncation=True, 
                max_length=max_length, 
                padding="max_length"
            )
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_masks[idx],
            "labels": self.input_ids[idx] # En GPT, los labels son el mismo input desplazado
        }

def entrenar():
    # 1. Cargar Tokenizer y Modelo
    print("Cargando modelo base...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    
    # GPT-2 no tiene token de padding por defecto, usamos el de fin de texto
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer)) # Ajustar por si hay tokens nuevos

    # 2. Preparar Datos
    raw_texts = cargar_datos_desde_json(DATA_PATH)
    if not raw_texts:
        print("No se encontraron datos. Revisa la ruta DATA_PATH.")
        return

    train_texts, val_texts = train_test_split(raw_texts, test_size=0.1)
    
    train_dataset = PoemasDataset(train_texts, tokenizer, MAX_LENGTH)
    val_dataset = PoemasDataset(val_texts, tokenizer, MAX_LENGTH)
    
    print(f"Entrenando con {len(train_dataset)} poemas.")

    # 3. Configuración del Entrenamiento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,              # Número de pasadas por los datos
        per_device_train_batch_size=4,   # Ajustar según tu memoria VRAM/RAM
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 4. Iniciar Entrenamiento
    trainer.train()
    
    # 5. Guardar el modelo final
    print("Guardando modelo final...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Modelo guardado en {OUTPUT_DIR}")

if __name__ == "__main__":
    entrenar()