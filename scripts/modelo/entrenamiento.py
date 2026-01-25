import json
import glob
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN "NO-OPENAI" ---
# Usamos GPT-Neo (EleutherAI) como principal. Es 100% Open Source.
MODEL_NAME_PREFERIDO = "EleutherAI/gpt-neo-125M" 
# Backup: BLOOM (BigScience), modelo multilingüe masivo open source.
MODEL_NAME_BACKUP = "bigscience/bloom-560m"

OUTPUT_DIR = "./modelo_poemas_neo_entrenado"
MAX_LENGTH = 256  # Longitud máxima del poema (tokens)

# --- REGLAS DE ESTRUCTURA (Prompt Engineering para el entrenamiento) ---
REGLAS_ESTRUCTURA = {
    # --- Estructuras Básicas ---
    "Pareado": "Escribe un Pareado: 2 versos que rimen entre sí (AA). Medida libre.",
    "Terceto": "Escribe un Terceto: 3 versos de 11 sílabas. Rima ABA.",
    "Terceto encadenado": "Escribe un Terceto Encadenado: Serie de tercetos. La rima del medio del primero es la rima de las puntas del siguiente (ABA BCB CDC...).",
    "Cuarteta": "Escribe una Cuarteta: 4 versos de 8 sílabas. Rima cruzada (abab).",
    "Redondilla": "Escribe una Redondilla: 4 versos de 8 sílabas. Rima abrazada (abba).",
    "Cuarteto": "Escribe un Cuarteto: 4 versos de 11 sílabas. Rima abrazada (ABBA).",
    "Serventesio": "Escribe un Serventesio: 4 versos de 11 sílabas. Rima cruzada (ABAB).",
    "Copla": "Escribe una Copla: 4 versos de 8 sílabas. Solo riman los pares (2 y 4). Impares sueltos.",
    "Seguidilla": "Escribe una Seguidilla: 4 versos. Sílabas: 7-5-7-5. Riman los pares en asonante.",

    # --- Estructuras Fijas ---
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
    
    # --- Formas Repetitivas ---
    "Villanelle": "Escribe una Villanelle: 19 versos (5 tercetos + 1 cuarteto). Solo dos rimas. Los versos 1 y 3 se repiten alternados.",
    "Sestina": "Escribe una Sestina (o sus 39 versos o un fragmento válido): 6 palabras finales se repiten en todas las estrofas en orden espiral.",
    "Pantoum": "Escribe un Pantoum: Cuartetos donde los versos 2 y 4 de una estrofa se convierten en el 1 y 3 de la siguiente.",
    "Rondeau": "Escribe un Rondeau: 15 versos. Rima aabba aabR aabbaR (R es la primera parte del primer verso usada como estribillo).",
    "Rondo": "Escribe un Rondó: Poema con estructura musical de repetición, similar al Rondeau pero con variaciones libres.",
    "Triolet": "Escribe un Triolet: 8 versos. Esquema ABaAabAB (Las mayúsculas son versos que se repiten idénticos).",
    "Limerick": "Escribe un Limerick: 5 versos de tono humorístico o absurdo. Rima AABBA. Ritmo anapéstico.",

    # --- Exóticas y Orientales ---
    "Haiku": "Escribe un Haiku: 3 versos. Sílabas 5-7-5. Captura un instante presente de la naturaleza (aquí y ahora). Sin rima.",
    "Tanka": "Escribe un Tanka: 5 versos. Sílabas 5-7-5-7-7. Empieza como imagen natural y termina con emoción profunda.",
    "ghazal": "Escribe un Ghazal (Gacela persa): Dísticos (pareados). Rima monorrima AA, bA, cA... con una palabra final repetida (Radif).",
    "Gacela": "Escribe una Gacela (estilo Lorquiano): Poema de intensidad erótica o mística, inspirada en la forma árabe pero con libertad métrica.",
    "Gacela_ghazal": "Escribe un Ghazal estricto: Mantén la estructura Qafiya (rima) y Radif (estribillo) al final de los versos pares.",

    # --- Visuales y Experimentales ---
    "Acrostico": "Escribe un Acróstico: Las letras iniciales de los versos deben leerse verticalmente formando la palabra clave.",
    "caligrama": "Escribe un Caligrama: El texto debe describir visualmente al objeto (Nota: genera el texto sugiriendo la forma).",
    "Palindromo poetico": "Escribe un Palíndromo o texto bifronte: Que pueda leerse igual de izquierda a derecha y viceversa (o frase a frase).",
    "Verso libre": "Escribe en Verso Libre: Sin métrica, ni rima, ni estrofa fija. Sigue el ritmo del pensamiento o la respiración.",
    "Versiculo": "Escribe en Versículos: Versos de gran extensión, ritmo majestuoso o profético (estilo bíblico o Whitman), sin rima.",

    # --- Géneros (El propósito) ---
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

    # --- Temáticas (Carpetas de tema) ---
    "Aventura_epica_heroismo": "Escribe un poema centrado en la Aventura, la Épica y el Heroísmo.",
    "Desamor_tristeza_perdida": "Escribe un poema profundo sobre Desamor, Tristeza y Pérdida.",
    "Religion_espiritualidad": "Escribe un poema místico sobre Religión y Espiritualidad.",
    "Sociedad_critica_social": "Escribe un poema de denuncia sobre la Sociedad y Crítica Social.",
    "Vida_y_existencia": "Escribe un poema filosófico sobre la Vida y la Existencia humana.",
    "Yo_interior_introspeccion": "Escribe un poema psicológico sobre el Yo Interior y la Introspección."
}

def obtener_regla(tipo_poema):
    tipo_limpio = tipo_poema.strip()
    return REGLAS_ESTRUCTURA.get(tipo_limpio, f"Escribe un poema de tipo: {tipo_limpio}")

# --- 1. CARGA DE DATOS ---
def cargar_datos_desde_json():
    textos = []
    base_path = os.getcwd()
    # Asumimos que la carpeta se llama 'datasets' (plural) según tus scripts anteriores
    search_path = os.path.join(base_path, "dataset_final_validado", "**", "*.json")
    
    print(f"📂 Buscando archivos en: {search_path}")
    files = glob.glob(search_path, recursive=True)
    print(f"✅ Encontrados {len(files)} archivos.")
    
    if len(files) == 0:
        print("⚠️ PRECAUCIÓN: No se encontraron archivos. Verifica la ruta 'datasets'.")

    count_validos = 0
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Acceso seguro a los datos
                poema_obj = data.get('poema', {})
                texto = poema_obj.get('texto', '')
                tipo = poema_obj.get('tipo') or data.get('subcategoria', 'Desconocido')
                keyword = poema_obj.get('palabra_clave_ingresada', '')
                
                # Solo agregamos si hay texto y palabra clave
                if texto and len(texto) > 10:
                    estructura = obtener_regla(tipo)
                    
                    # Formato de entrenamiento (Prompt -> Output)
                    # Usamos tokens estándar que GPT-Neo entiende bien
                    formato_entrenamiento = (
                        f"TIPO: {tipo}\n"
                        f"INSTRUCCION: {estructura}\n"
                        f"TEMA: {keyword}\n"
                        f"### POEMA:\n{texto}"
                        f"<|endoftext|>"
                    )
                    textos.append(formato_entrenamiento)
                    count_validos += 1
        except Exception as e:
            # print(f"Error en {file}: {e}") # Descomentar para debug
            pass
    
    print(f"📖 Textos válidos procesados: {count_validos}")     
    return textos

# --- 2. DATASET PERSONALIZADO ---
class PoemasDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        
        print("⚙️ Tokenizando dataset... (esto puede tardar unos segundos)")
        for txt in txt_list:
            # truncation=True corta si es muy largo, padding=True rellena si es corto
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
    print("\n--- INICIANDO SISTEMA DE ENTRENAMIENTO (ELEUTHER AI) ---")
    
    # 1. SETUP DISPOSITIVO
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 GPU DETECTADA: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ ALERTA: Usando CPU. El entrenamiento será lento.")

    # 2. CARGA DE MODELO
    modelo_actual = MODEL_NAME_PREFERIDO
    try:
        print(f"⬇️ Descargando/Cargando {modelo_actual}...")
        # AutoTokenizer es clave para soportar GPT-Neo/BLOOM
        tokenizer = AutoTokenizer.from_pretrained(modelo_actual)
        model = AutoModelForCausalLM.from_pretrained(modelo_actual)
    except Exception as e:
        print(f"⚠️ Falló {modelo_actual} ({e}), intentando backup...")
        modelo_actual = MODEL_NAME_BACKUP
        tokenizer = AutoTokenizer.from_pretrained(modelo_actual)
        model = AutoModelForCausalLM.from_pretrained(modelo_actual)
    
    model.to(device)
    
    # IMPORTANTE: GPT-Neo no tiene pad_token por defecto, hay que asignarlo
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("ℹ️ Pad token configurado como EOS token.")

    # 3. PREPARAR DATOS
    raw_texts = cargar_datos_desde_json()
    if not raw_texts:
        print("❌ ERROR CRÍTICO: No hay datos para entrenar.")
        return

    # Dividir Train/Test
    train_texts, val_texts = train_test_split(raw_texts, test_size=0.1)
    
    train_dataset = PoemasDataset(train_texts, tokenizer, MAX_LENGTH)
    val_dataset = PoemasDataset(val_texts, tokenizer, MAX_LENGTH)
    
    print(f"📚 Dataset listo: {len(train_dataset)} entrenamiento | {len(val_dataset)} validación")

    # 4. CONFIGURACIÓN DEL TRAINER
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=6,              # Pasadas completas al dataset
        per_device_train_batch_size=4,   # Bajado a 4 por seguridad (GPT-Neo consume RAM)
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,   # Simula un batch de 8 (4x2)
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,              # Solo guarda los 2 últimos checkpoints para no llenar disco
        fp16=torch.cuda.is_available(),  # Usa media precisión si hay GPU (ahorra memoria)
        dataloader_num_workers=0,        # 0 para evitar problemas en Windows
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 5. EJECUTAR
    print("🔥 ¡Comenzando entrenamiento! Ve por un café...")
    trainer.train()
    
    # 6. GUARDAR
    print("💾 Guardando modelo final...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ ¡Éxito! Modelo guardado en: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    entrenar()