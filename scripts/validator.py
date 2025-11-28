import json
import os
import hashlib
from jsonschema import validate, ValidationError

SCHEMA_PATH = "schema/poema.schema.json"
DATASET_PATH = "datasets"
HASH_REGISTRY = "scripts/hash_registry.txt"


def hash_poema(texto: str) -> str:
    """Crea un hash único basado en el poema para detectar duplicados."""
    return hashlib.sha256(texto.strip().encode("utf-8")).hexdigest()


def load_hashes() -> set:
    """Carga los hashes registrados previamente para identificar duplicados."""
    if not os.path.exists(HASH_REGISTRY):
        return set()
    with open(HASH_REGISTRY, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())


def save_hash(h: str) -> None:
    """Guarda un hash nuevo en el archivo de registro."""
    with open(HASH_REGISTRY, "a", encoding="utf-8") as f:
        f.write(h + "\n")


def validate_json(data: dict, path: str) -> None:
    """Valida que el JSON cumpla con el schema."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        raise Exception(f"❌ ERROR EN FORMATO en {path}\n→ {e.message}")


def scan_datasets():
    stored_hashes = load_hashes()
    new_hashes = set()

    print("🔍 Escaneando dataset...\n")

    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            if not file.endswith(".json"):
                continue

            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            validate_json(data, path)

            poema_texto = data["poema"]["texto"]
            poema_hash = hash_poema(poema_texto)

            if poema_hash in stored_hashes or poema_hash in new_hashes:
                raise Exception(f"🚫 DUPLICADO DETECTADO en:\n{path}\n"
                                f"Este poema ya existe en el dataset.")
            else:
                new_hashes.add(poema_hash)

    for h in new_hashes:
        save_hash(h)

    print("✅ VALIDACIÓN EXITOSA — sin duplicados y con formato correcto")


if __name__ == "__main__":
    scan_datasets()
