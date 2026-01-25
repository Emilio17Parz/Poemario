import json
import glob
import os
import torch
import random
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
MODEL_NAME_PREFERIDO = "EleutherAI/gpt-neo-125M"
OUTPUT_DIR = "./modelo_poemas_neo_balanceado_v3"
MAX_LENGTH = 256

# --- CONFIGURACIÓN DEL ENTRENAMIENTO ---
NUM_EPOCHS = 50
# Ajustado a 3400 para igualar a tu categoría más pequeña (Villanelle ~3494)
# Esto garantiza un balanceo matemático perfecto.
SAMPLES_POR_CLASE = 3400 

# --- REGLAS DE ESTRUCTURA ---
REGLAS_ESTRUCTURA = {
    "Pareado": "Escribe un Pareado: 2 versos que rimen entre sí (AA). Medida libre.",
    "Terceto": "Escribe un Terceto: 3 versos de 11 sílabas. Rima ABA.",
    "Terceto encadenado": "Escribe un Terceto Encadenado: Serie de tercetos. La rima del medio del primero es la rima de las puntas del siguiente (ABA BCB CDC...).",
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

def obtener_regla(tipo_poema):
    tipo_limpio = tipo_poema.strip()
    return REGLAS_ESTRUCTURA.get(tipo_limpio, f"Escribe un poema de tipo: {tipo_limpio}")

def cargar_datos_balanceados():
    textos = []
    base_path = os.getcwd()
    dataset_path = os.path.join(base_path, "dataset_final_validado")
    
    print(f" Buscando categorías en: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(" ERROR: No se encuentra la carpeta 'dataset_final_validado'.")
        return []

    categorias = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f" Categorías encontradas: {len(categorias)}")

    total_procesados = 0

    for categoria in categorias:
        cat_path = os.path.join(dataset_path, categoria)
        archivos = glob.glob(os.path.join(cat_path, "*.json"))
        cantidad_total = len(archivos)
        
        if cantidad_total == 0:
            continue
            
        # Balanceo: si hay más de 3400, toma 3400 random. Si hay menos, toma todos.
        if cantidad_total > SAMPLES_POR_CLASE:
            archivos_seleccionados = random.sample(archivos, SAMPLES_POR_CLASE)
        else:
            archivos_seleccionados = archivos 
        
        print(f"   🔹 {categoria}: {cantidad_total} -> Usando {len(archivos_seleccionados)}")

        for file in archivos_seleccionados:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    poema_obj = data.get('poema', {})
                    texto = poema_obj.get('texto', '')
                    tipo = categoria 
                    keyword = poema_obj.get('palabra_clave_ingresada', 'tema libre')
                    
                    if texto and len(texto) > 10:
                        estructura = obtener_regla(tipo)
                        formato_entrenamiento = (
                            f"TIPO: {tipo}\n"
                            f"INSTRUCCION: {estructura}\n"
                            f"TEMA: {keyword}\n"
                            f"### POEMA:\n{texto}"
                            f"<|endoftext|>"
                        )
                        textos.append(formato_entrenamiento)
                        total_procesados += 1
            except Exception:
                pass
    
    print(f" Total final de textos para entrenamiento: {total_procesados}")
    return textos

class PoemasDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        print(f"⚙️ Tokenizando {len(txt_list)} textos... (Paciencia, esto usa RAM)")
        for txt in txt_list:
            encodings_dict = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx], 
            "attention_mask": self.attn_masks[idx], 
            "labels": self.input_ids[idx]
        }

def entrenar():
    print("\n--- ENTRENAMIENTO BALANCEADO (50 ÉPOCAS - FIXED) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Dispositivo: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_PREFERIDO)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_PREFERIDO)
    model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_texts = cargar_datos_balanceados()
    if not raw_texts:
        return

    # Dividimos 95% Train - 5% Val (suficiente para tantos datos)
    train_texts, val_texts = train_test_split(raw_texts, test_size=0.05)
    
    train_dataset = PoemasDataset(train_texts, tokenizer, MAX_LENGTH)
    val_dataset = PoemasDataset(val_texts, tokenizer, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=4,   
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        
        # --- CORRECCIÓN CRÍTICA AQUÍ ---
        eval_strategy="epoch",      # Antes: evaluation_strategy (deprecated)
        save_strategy="epoch",      # Guarda cada época
        
        load_best_model_at_end=True, # Al final, carga la mejor época (menor loss)
        metric_for_best_model="loss",
        greater_is_better=False,
        
        save_total_limit=2,         # Guarda solo los 2 últimos checkpoints para no llenar disco
        logging_dir='./logs',
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        # Opcional: Detiene si no mejora en 3 épocas seguidas (ahorra tiempo)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )

    print(" Iniciando entrenamiento... (Si no mejora en 3 épocas, parará antes)")
    trainer.train()
    
    print(" Guardando el MEJOR modelo encontrado...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f" ¡Modelo guardado en {OUTPUT_DIR}!")

if __name__ == "__main__":
    entrenar()