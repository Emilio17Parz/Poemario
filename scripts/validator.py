import json
import os
import hashlib
from jsonschema import validate, ValidationError

SCHEMA_PATH = "schema/poema.schema.json"
DATASET_PATH = "datasets"
HASH_REGISTRY = "scripts/hash_registry.txt"


def hash_poema(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def load_hashes():
    if not os.path.exists(HASH_REGISTRY):
        return set()
    with open(HASH_REGISTRY, "r") as f:
        return set(line.strip() for line in f.readlines())


def save_hash(h):
    with open(HASH_REGISTRY, "a") as f:
        f.write(h + "\n")


def validate_json(data):
    with open(SCHEMA_PATH) as schema_file:
        schema = json.load(schema_file)
        validate(instance=data, schema=schema)


def scan_datasets():
    stored_hashes = load_hashes()

    for folder, _, files in os.walk(DATASET_PATH):
        for file in files:
            if not file.endswith(".json"):
                continue

            path = os.path.join(folder, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            try:
                validate_json(data)
            except ValidationError as e:
                raise Exception(f"❌ ERROR EN FORMATO en {path} → {e.message}")

            poema_hash = hash_poema(data["poema"]["texto"])

            if poema_hash in stored_hashes:
                raise Exception(f"❌ DUPLICADO detectado en {path}")
            else:
                save_hash(poema_hash)

    print("✅ VALIDACIÓN EXITOSA — sin duplicados y con formato correcto")


if __name__ == "__main__":
    scan_datasets()
