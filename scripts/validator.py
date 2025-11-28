import json
import os
import hashlib
from jsonschema import validate, ValidationError

SCHEMA_PATH = "schema/poema.schema.json"
DATASET_PATH = "datasets"
HASH_REGISTRY = "scripts/hash_registry.txt"


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

    hashes = {}  # hash → archivo original
    duplicados = []

    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            if not file.endswith(".json"):
                continue

            path = os.path.join(root, file)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            validate_json(data, path)

            texto = data["poema"]["texto"]
            h = hash_poema(texto)

            if h in hashes:
                duplicados.append((path, hashes[h]))
            else:
                hashes[h] = path

    if duplicados:
        msg = "🚫 SE DETECTARON POEMAS DUPLICADOS:\n"
        for p1, p2 in duplicados:
            msg += f"\t❌ {p1} es idéntico a {p2}\n"
        raise Exception(msg)

    print("✅ VALIDACIÓN EXITOSA — no hay duplicados ni errores de formato")


if __name__ == "__main__":
    scan_datasets()
