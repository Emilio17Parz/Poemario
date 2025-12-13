import os
import json
import time
import random
import re
from openai import OpenAI
from openai import RateLimitError

# =============================
# CONFIGURACIÓN GENERAL
# =============================

BASE_DATASET_PATH = "datasets"
TIPO_POEMA_GENERAR = "Acrostico"   # ← TIPO FIJO
MODELO = "gpt-4o-mini"

TAMANO_LOTE = 60
ESPERA_ENTRE_LLAMADAS = 5
ESPERA_ENTRE_LOTES = 20

# =============================
# CLIENTE OPENAI
# =============================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================
# UTILIDADES
# =============================

def limpiar_nombre(texto: str) -> str:
    texto = texto.lower().strip()
    texto = texto.replace(" ", "_")
    texto = "".join(c for c in texto if c.isalnum() or c == "_")
    return texto[:60]

def limpiar_tema(tema: str) -> str:
    tema = tema.lower().strip()
    tema = "".join(c for c in tema if c.isalpha())
    return tema

def cargar_existentes(tipo):
    ruta = os.path.join(BASE_DATASET_PATH, tipo)
    existentes = {"textos": set(), "subcategorias": set()}
    os.makedirs(ruta, exist_ok=True)

    for archivo in os.listdir(ruta):
        if archivo.endswith(".json"):
            try:
                with open(os.path.join(ruta, archivo), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "poema" in data:
                        existentes["textos"].add(data["poema"]["texto"].strip())
                    if "subcategoria" in data:
                        existentes["subcategorias"].add(data["subcategoria"].strip().lower())
            except:
                pass
    return existentes

# =============================
# PALABRA CLAVE
# =============================

def extraer_palabra_clave(subcategoria: str, texto_poema: str) -> str:
    # En acróstico, la palabra clave ES el tema
    return subcategoria.capitalize()

# =============================
# TEMA ESPECÍFICO (ACRÓSTICO)
# =============================

def generar_tema_acrostico(temas_existentes: set):
    temas_previos = ", ".join(list(temas_existentes)[:15])

    prompt = f"""
Genera UNA SOLA PALABRA en español para un ACRÓSTICO.

REGLAS ESTRICTAS:
- SOLO una palabra (sin espacios)
- Longitud entre 5 y 9 letras
- Letras simples (sin acentos, sin ñ)
- Sustantivo o concepto claro
- Tema diverso (urbano, abstracto, científico, simbólico, cotidiano, tecnológico)

NO debe parecerse a estas palabras ya usadas:
{temas_previos}

Ejemplos válidos (NO reutilizar):
"silencio"
"rutina"
"archivo"
"memoria"
"frontera"
"latido"

NO expliques nada.
NO comillas.
Devuelve SOLO la palabra.
"""

    r = client.chat.completions.create(
        model=MODELO,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.1
    )

    return limpiar_tema(r.choices[0].message.content)

# =============================
# PROMPT ACRÓSTICO
# =============================

def construir_prompt_acrostico(tema):
    letras = list(tema.upper())

    letras_str = ", ".join(letras)

    return f"""
Genera un ACRÓSTICO ORIGINAL en español.

PALABRA CLAVE: {tema.upper()}

REGLAS OBLIGATORIAS E INNEGOCIABLES:
- Número de versos: EXACTAMENTE {len(letras)}
- Cada verso debe comenzar EXACTAMENTE con estas letras, en este orden:
  {letras_str}
- La letra inicial debe ser visible (primera letra del verso)
- Verso libre
- Lenguaje claro y coherente
- El contenido debe relacionarse con el tema

PROHIBIDO:
- Títulos
- Explicaciones
- Cambiar el orden de las letras
- Añadir versos extra

Devuelve SOLO el poema con saltos de línea.
"""

# =============================
# GPT
# =============================

def llamar_gpt(prompt: str) -> str:
    r = client.chat.completions.create(
        model=MODELO,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content.strip()

# =============================
# VALIDACIÓN ACRÓSTICO
# =============================

def validar_acrostico(texto: str, tema: str) -> bool:
    versos = [v for v in texto.split("\n") if v.strip()]
    if len(versos) != len(tema):
        return False

    for verso, letra in zip(versos, tema.upper()):
        if not verso.strip().startswith(letra):
            return False

    return True

# =============================
# GUARDAR
# =============================

def guardar_poema(tipo, subcategoria, palabra_clave, texto):
    carpeta = os.path.join(BASE_DATASET_PATH, tipo)
    os.makedirs(carpeta, exist_ok=True)

    ruta = os.path.join(carpeta, limpiar_nombre(subcategoria) + ".json")

    data = {
        "subcategoria": subcategoria,
        "poema": {
            "texto": texto.strip(),
            "tipo": tipo,
            "palabra_clave_ingresada": palabra_clave
        }
    }

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Guardado: {ruta}")

# =============================
# MAIN
# =============================

def main():
    print(f"\n📜 GENERADOR EXCLUSIVO DE ACRÓSTICOS 📜\n")

    total = int(input("¿Cuántos acrósticos deseas generar?: "))

    existentes = cargar_existentes(TIPO_POEMA_GENERAR)
    temas_existentes = existentes["subcategorias"]
    textos_existentes = existentes["textos"]

    contador = 0
    MAX_REINTENTOS_TEMA = 6

    while contador < total:
        lote = 0
        print(f"\n--- LOTE {contador//TAMANO_LOTE + 1} ---\n")

        while lote < TAMANO_LOTE and contador < total:
            try:
                tema = None
                intentos = 0

                while tema is None or tema in temas_existentes:
                    if intentos >= MAX_REINTENTOS_TEMA:
                        tema = generar_tema_acrostico(set())
                        break
                    tema = generar_tema_acrostico(temas_existentes)
                    intentos += 1

                texto = llamar_gpt(construir_prompt_acrostico(tema))

                if not validar_acrostico(texto, tema):
                    print("⚠️ Acróstico inválido. Regenerando...")
                    continue

                if texto in textos_existentes:
                    continue

                palabra = extraer_palabra_clave(tema, texto)

                guardar_poema(TIPO_POEMA_GENERAR, tema, palabra, texto)

                textos_existentes.add(texto)
                temas_existentes.add(tema)

                contador += 1
                lote += 1
                time.sleep(ESPERA_ENTRE_LLAMADAS)

            except RateLimitError:
                print("⏳ Rate limit. Esperando...")
                time.sleep(ESPERA_ENTRE_LOTES)

        if contador < total:
            time.sleep(ESPERA_ENTRE_LOTES)

    print(f"\n✨ Generación finalizada: {total} acrósticos.\n")

# =============================

if __name__ == "__main__":
    main()
