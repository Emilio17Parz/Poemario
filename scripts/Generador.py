import os
import json
import time
import random
import re
from openai import OpenAI

# =============================
# CONFIGURACIÓN GENERAL
# =============================

BASE_DATASET_PATH = "datasets"
MODELO = "gpt-4o-mini"   # RECOMENDADO

TAMANO_LOTE = 50
ESPERA_ENTRE_LLAMADAS = 5
ESPERA_ENTRE_LOTES = 15

# =============================
# CLIENTE OPENAI
# =============================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================
# TIPOS DISPONIBLES
# =============================

def obtener_tipos():
    return sorted([
        d for d in os.listdir(BASE_DATASET_PATH)
        if os.path.isdir(os.path.join(BASE_DATASET_PATH, d))
    ])

# =============================
# UTILIDADES
# =============================

def limpiar_nombre(texto: str) -> str:
    texto = texto.lower().strip()
    texto = texto.replace(" ", "_")
    texto = "".join(c for c in texto if c.isalnum() or c == "_")
    return texto[:60]

def limpiar_tema(tema: str) -> str:
    tema = tema.replace('"', "").replace("'", "")
    tema = " ".join(tema.replace("\n", " ").split())
    return tema.strip()

def cargar_existentes(tipo):
    ruta = os.path.join(BASE_DATASET_PATH, tipo)
    existentes = set()

    if os.path.exists(ruta):
        for archivo in os.listdir(ruta):
            if archivo.endswith(".json"):
                try:
                    with open(os.path.join(ruta, archivo), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "poema" in data and "texto" in data["poema"]:
                             existentes.add(data["poema"]["texto"].strip())
                except json.JSONDecodeError:
                    print(f"⚠️ Error al decodificar JSON en: {archivo}")
    return existentes

# =============================
# EXTRAER PALABRA CLAVE
# =============================

def extraer_palabra_clave(subcategoria: str, texto_poema: str) -> str:
    prompt = f"""
Analiza el siguiente poema cuyo tema era '{subcategoria}'.
Devuelve SOLO UNA palabra que sea el concepto o sustantivo MÁS relevante del texto.
Debe ser un sustantivo. NO artículos, NO preposiciones.

Poema:
---
{texto_poema}
---

SOLO la palabra.
NO comillas.
NO explicaciones.
"""

    try:
        response = client.responses.create(
            model=MODELO,
            input=prompt
        )
        palabra = response.output_text.strip().split()[0]
        palabra = "".join(c for c in palabra if c.isalpha())
        return palabra.capitalize()

    except Exception as e:
        print(f"⚠️ Error extrayendo palabra clave: {e}")
        try:
            return subcategoria.split()[0].capitalize()
        except:
            return "Tema"

# =============================
# TEMA ALEATORIO
# =============================

def generar_tema_aleatorio():
    prompt = """
Genera un TEMA para un poema en español.
Debe ser UNA sola palabra O UNA frase corta de máximo 4 palabras.
NO USES LA MISMA PALBRA O TEMA QUE YA HAYAS USADO ANTERIORMENTE AL INICIO DEL SIGUIENTE TEMA.
ESTOS DEBEN SER ABSURDOS, INUSUALES, CREATIVOS. 
REQUISITOS:
- No uses palabras comunes como: amor, CUERPOS CELEXTES, agujero, vida, muerte, viento, silencio, luz, sombra, ECOS,ecos, sueños, galaxia.
- No uses temas repetidos dentro de esta misma respuesta.
- Debe sonar creativo, inusual, evocador o simbólico.
- Varía entre: científico, histórico, surrealista, rural, urbano, mítico, tecnológico, absurdo, filosófico.
- Evita repetir estructuras como: "El ..." o "La ...".
- NO expliques nada.
- NO incluyas comillas.

Devuelve SOLO el tema.
"""

    response = client.responses.create(
        model=MODELO,
        input=prompt
    )

    return limpiar_tema(response.output_text)


# =============================
# PROMPTS POÉTICOS
# =============================

def construir_prompt(tipo, tema):
    if tipo == "Villanelle":
        return f"""
Genera una VILLANELLE ORIGINAL en español.

Tema: {tema}

REGLAS OBLIGATORIAS:
- 5 tercetos + 1 cuarteto final (19 versos exactos).
- Estribillos A1 y A2 repetidos exactamente.
-Debes de cumplir la métrica y rima estrictamente.
- Sin título, sin explicación.
"""

    if tipo == "Romance":
        return f"""
Genera un ROMANCE NARRATIVO.

Tema: {tema}

Reglas:
- Versos octosílabos.
- Versos impares sin rima.
- Versos pares con rima asonante constante.
- Sin título ni explicación.
"""

    if tipo == "Haiku":
        return f"""
Genera un HAIKU sobre: {tema}

Reglas:
- 3 versos exactos.
- 5 / 7 / 5 sílabas.
- Naturaleza, sin rima, sin título.
"""

    if tipo == "Soneto":
        return f"""
Genera un SONETO clásico sobre: {tema}

Reglas:
- 14 versos endecasílabos.
- Rima consonante.
- 2 cuartetos + 2 tercetos.
- Sin título ni explicación.
"""

    return f"""
Genera un poema del tipo {tipo} sobre: {tema}
Respeta estrictamente su MÉTRICA EN CADA VERSO, NUMERO DE VERSOS QUE CORRESPONDAN AL TIPO DE POEMA, rima y estructura real.
Sin título.
Sin explicación.
"""

# =============================
# GENERAR TEXTO (GPT)
# =============================

def llamar_gpt(prompt: str) -> str:
    response = client.responses.create(
        model=MODELO,
        input=prompt
    )
    return response.output_text.strip()

# =============================
# GUARDAR
# =============================

def guardar_poema(tipo, subcategoria, palabra_clave, texto):
    carpeta = os.path.join(BASE_DATASET_PATH, tipo)
    os.makedirs(carpeta, exist_ok=True)

    nombre_archivo = limpiar_nombre(subcategoria) + ".json"
    ruta = os.path.join(carpeta, nombre_archivo)

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
    print("\n📜 GENERADOR MASIVO UNIVERSAL (OpenAI GPT-4O-MINI) 📜\n")

    tipos = obtener_tipos()
    for i, t in enumerate(tipos):
        print(f"{i+1}. {t}")

    try:
        seleccion = int(input("\nSelecciona el tipo de poema: ")) - 1
        tipo_elegido = tipos[seleccion]
    except:
        print("Selección inválida.")
        return

    try:
        total = int(input("¿Cuántos poemas deseas generar?: "))
    except:
        print("Número inválido.")
        return

    existentes = cargar_existentes(tipo_elegido)

    print(f"\n🎯 Generando {total} poemas tipo {tipo_elegido}...\n")

    contador = 0

    while contador < total:
        lote_actual = 0
        print(f"\n--- LOTE {int(contador/TAMANO_LOTE)+1} ---\n")

        while lote_actual < TAMANO_LOTE and contador < total:
            try:
                tema = generar_tema_aleatorio()
                subcategoria = tema

                prompt = construir_prompt(tipo_elegido, tema)
                texto = llamar_gpt(prompt)

                palabra = extraer_palabra_clave(subcategoria, texto)

                if not texto.strip():
                    print("⚠️ Vacío, regenerando...")
                    continue

                if texto.strip() in existentes:
                    print("⚠️ Duplicado, regenerando...")
                    continue

                guardar_poema(tipo_elegido, subcategoria, palabra, texto)

                existentes.add(texto.strip())
                contador += 1
                lote_actual += 1

                time.sleep(ESPERA_ENTRE_LLAMADAS)

            except Exception as e:
                print(f"❌ Error: {e}")
                print("⏳ Esperando para reintentar...")
                time.sleep(10)
                continue

        if contador < total:
            print(f"\n💤 Descanso de lote ({ESPERA_ENTRE_LOTES}s)...\n")
            time.sleep(ESPERA_ENTRE_LOTES)

    print(f"\n✨ Finalizada la generación de {total} poemas.\n")

# =============================
# EJECUCIÓN
# =============================

if __name__ == "__main__":
    main()
