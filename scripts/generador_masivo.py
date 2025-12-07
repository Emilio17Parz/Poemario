import os
import json
import time
import random
import requests

# =============================
# CONFIGURACIÓN GENERAL
# =============================

BASE_DATASET_PATH = "datasets"
TIPO_FIJO = "Romance"   # 🔒 SOLO para esta categoría
ESPERA_ENTRE_LLAMADAS = 1.0
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO = "mistral"  # o "llama3"

# =============================
# UTILIDADES
# =============================

def limpiar_nombre(texto: str) -> str:
    texto = texto.lower().strip()
    texto = texto.replace(" ", "_")
    texto = "".join(c for c in texto if c.isalnum() or c == "_")
    return texto[:50]

def limpiar_tema(tema: str) -> str:
    tema = tema.replace('"', "").replace("'", "")
    tema = tema.replace("\n", " ").replace("\r", " ")
    tema = " ".join(tema.split())
    return tema.strip()

def cargar_existentes():
    ruta = os.path.join(BASE_DATASET_PATH, TIPO_FIJO)
    existentes = set()

    if os.path.exists(ruta):
        for archivo in os.listdir(ruta):
            if archivo.endswith(".json"):
                with open(os.path.join(ruta, archivo), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    existentes.add(data["poema"]["texto"].strip())

    return existentes

# =============================
# GENERAR TEMA NO CLICHÉ
# =============================

def generar_tema_aleatorio():
    prompt = """
Devuelve SOLO una palabra o frase corta como tema para un romance narrativo en español.

Debe ser:
- Poco común
- Puede ser histórico, trágico, simbólico, fantástico o cotidiano
- Evita clichés: amor, luna, mar, noche, estrellas, dolor

NO expliques nada.
NO agregues texto adicional.
"""
    respuesta = llamar_ollama(prompt)
    return respuesta.strip()

# =============================
# PROMPT ESTRICTO PARA ROMANCE
# =============================

def construir_prompt_romance(tema: str):
    versos = random.randint(14, 20)

    return f"""
Genera un ROMANCE ORIGINAL en español.

Tema central: {tema}
Número de versos: EXACTAMENTE {versos}

REGLAS MÉTRICAS OBLIGATORIAS (NO IGNORAR):

1. TODOS los versos deben ser OCTOSÍLABOS (8 sílabas exactas).
2. Los versos IMPARES:
   - NO deben rimar.
3. Los versos PARES:
   - DEBEN rimar entre sí con rima ASONANTE.
   - La coincidencia debe ser VOCÁLICA desde la última vocal tónica.
4. El poema DEBE ser un SOLO BLOQUE CONTINUO:
   - NO dividir en estrofas.
   - NO separar en pareados.
5. El tono DEBE ser NARRATIVO:
   - Debe contar una historia.
   - No debe ser introspectivo ni lírico.
6. Lenguaje poético pero claro.
7. NO incluir títulos.
8. NO incluir explicaciones.
9. NO incluir frases como "Aquí tienes".
10. NO repetir versos.
11. Devuelve ÚNICAMENTE el poema en versos, sin encabezados ni comentarios.

SI NO CUMPLES LA MÉTRICA Y LA RIMA, EL POEMA ES INVÁLIDO.
"""

    versos = random.randint(12, 20)

    return f"""
Genera un ROMANCE ORIGINAL en español.

Tema central: {tema}
Número de versos: {versos}

REGLAS OBLIGATORIAS DEL ROMANCE:
- Versos OCTOSÍLABOS
- Versos IMPARES sin rima
- Versos PARES con RIMA ASONANTE entre sí
- Tono NARRATIVO
- Debe contar una historia (no introspectivo)
- NO debe ser lírico, satírico ni épico
- Lenguaje claro, poético y narrativo
- NO incluir títulos
- NO incluir explicaciones
- NO incluir frases como "Aquí tienes"
- NO repetir versos
- Devuelve ÚNICAMENTE el poema en versos
"""

# =============================
# LLAMAR A OLLAMA
# =============================

def llamar_ollama(prompt: str) -> str:
    payload = {
        "model": MODELO,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()
    return data["response"]

# =============================
# GUARDAR POEMA
# =============================

def guardar_poema(subcategoria, palabra_clave, texto):
    carpeta = os.path.join(BASE_DATASET_PATH, TIPO_FIJO)
    os.makedirs(carpeta, exist_ok=True)

    nombre_archivo = limpiar_nombre(subcategoria) + ".json"
    ruta = os.path.join(carpeta, nombre_archivo)

    data = {
        "subcategoria": subcategoria,
        "poema": {
            "texto": texto.strip(),
            "tipo": TIPO_FIJO,
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
    print("🚀 GENERADOR MASIVO — ROMANCE (ESTRUCTURA REAL)\n")

    total = int(input("¿Cuántos romances deseas generar?: "))

    existentes = cargar_existentes()

    print(f"\n🎯 Generando {total} romances...\n")

    contador = 0

    while contador < total:
        try:
            tema = generar_tema_aleatorio()
            tema = limpiar_tema(tema)

            subcategoria = tema
            palabra_clave = tema.split()[0]

            prompt_poema = construir_prompt_romance(tema)
            texto = llamar_ollama(prompt_poema)

            if not texto.strip():
                print("⚠️ Romance vacío, regenerando...")
                continue

            if texto.strip() in existentes:
                print("⚠️ Romance duplicado, regenerando...")
                continue

            guardar_poema(subcategoria, palabra_clave, texto)
            existentes.add(texto.strip())
            contador += 1

            time.sleep(ESPERA_ENTRE_LLAMADAS)

        except Exception as e:
            print(f"❌ Error generando romance: {e}")
            time.sleep(2)

    print("\n✅ Generación finalizada de ROMANCES.\n")
    print("🟡 Validador se aplicará en el Pull Request")

if __name__ == "__main__":
    main()
