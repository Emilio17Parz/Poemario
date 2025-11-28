import json
import os
import hashlib
from jsonschema import validate, ValidationError

SCHEMA_PATH = "schema/poema.schema.json"
DATASET_PATH = "datasets"

def hash_poema(texto: str) -> str:
    return hashlib.sha256(texto.strip().encode("utf-8")).hexdigest()

def validate_json(data, path):
    with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        raise Exception(f"❌ ERROR EN FORMATO en {path}\n→ {e.message}")

def scan_datasets():
    print("🔍 Escaneando dataset...\n")

    hashes = {}       # hash -> archivo original
    duplicados = []   # lista de conflictos

    for ruta, _, archivos in os.walk(DATASET_PATH):
        for archivo in archivos:
            if archivo.endswith(".json"):
                path = os.path.join(ruta, archivo)

                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                validate_json(data, path)
                h = hash_poema(data["poema"]["texto"])

                if h in hashes:
                    duplicados.append((path, hashes[h]))
                else:
                    hashes[h] = path

    if duplicados:
        msg = "🚫 SE DETECTARON POEMAS DUPLICADOS:\n"
        for p1, p2 in duplicados:
            msg += f"❌ {p1} es idéntico a {p2}\n"
        raise Exception(msg)

    print("✅ VALIDACIÓN EXITOSA — no hay duplicados ni errores")

if __name__ == "__main__":
    scan_datasets()
