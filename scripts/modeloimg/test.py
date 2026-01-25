import os
import json
import argparse
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

DEFAULT_SAVE_PATH = "saved_models_final"
DEFAULT_MODEL_NAME = "gen_final.keras"
DEFAULT_OUTDIR = "outputs_gen"
DEFAULT_LATENT_DIM = 128
DEFAULT_IMG_SIZE = 64

# ----------------------------
# Reproducibilidad
# ----------------------------
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def denorm_img(x):
    x = (x + 1.0) * 0.5
    x = np.clip(x, 0.0, 1.0)
    return (x * 255).astype(np.uint8)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def truncation_trick(z, psi=0.7):
    return z * psi

# ----------------------------
# Cargar modelo + encoders
# ----------------------------
def load_assets(save_path, model_name):
    gen_path = os.path.join(save_path, model_name)
    enc_con_path = os.path.join(save_path, "encoder_con.pkl")
    enc_sub_path = os.path.join(save_path, "encoder_sub.pkl")

    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"No encontré el generator en: {gen_path}")
    if not os.path.exists(enc_con_path):
        raise FileNotFoundError(f"No encontré encoder_con.pkl en: {enc_con_path}")
    if not os.path.exists(enc_sub_path):
        raise FileNotFoundError(f"No encontré encoder_sub.pkl en: {enc_sub_path}")

    gen = tf.keras.models.load_model(gen_path, compile=False)
    le_con = joblib.load(enc_con_path)
    le_sub = joblib.load(enc_sub_path)
    return gen, le_con, le_sub

def label_to_idx(le, label: str, kind="label"):
    classes = list(le.classes_)
    if label not in classes:
        example = ", ".join(classes[:40])
        raise ValueError(
            f"{kind} '{label}' no existe en el encoder. Ejemplos: {example} ..."
        )
    return int(le.transform([label])[0])

# ----------------------------
# Leer estructura del dataset
# datasetimg/
#   ConceptoA/
#       Sub1/
#       Sub2/
#   ConceptoB/
#       (si no hay subcarpetas, se usa el mismo)
# ----------------------------
def scan_dataset_structure(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No existe dataset_path: {dataset_path}")

    concepts = {}
    for concept in sorted(os.listdir(dataset_path)):
        cpath = os.path.join(dataset_path, concept)
        if not os.path.isdir(cpath):
            continue

        subs = []
        # Subcarpetas
        for sub in sorted(os.listdir(cpath)):
            spath = os.path.join(cpath, sub)
            if os.path.isdir(spath):
                subs.append(sub)

        # Si no hay subcarpetas, el subconcepto = concepto
        if not subs:
            subs = [concept]

        concepts[concept] = subs

    if not concepts:
        raise RuntimeError(f"No encontré carpetas de conceptos dentro de: {dataset_path}")
    return concepts

def print_catalog(catalog, limit_subs=30):
    print("📚 Catálogo (datasetimg):")
    for c, subs in catalog.items():
        shown = subs[:limit_subs]
        tail = "" if len(subs) <= limit_subs else f" ... (+{len(subs)-limit_subs})"
        print(f"- {c}: {', '.join(shown)}{tail}")

# ----------------------------
# Seeds fijas por etiqueta
# ----------------------------
def get_fixed_seed_map(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_fixed_seed_map(path, seed_map):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(seed_map, f, ensure_ascii=False, indent=2)

def make_key(concept, subconcept):
    return f"{concept}::{subconcept}"

# ----------------------------
# Generación
# ----------------------------
def generate_images(gen, con_idx, sub_idx, n, latent_dim, seed, psi):
    set_global_seed(seed)
    z = np.random.normal(0, 1, size=(n, latent_dim)).astype(np.float32)
    z = truncation_trick(z, psi=psi).astype(np.float32)

    c = np.full((n, 1), con_idx, dtype=np.int32)
    s = np.full((n, 1), sub_idx, dtype=np.int32)

    imgs = gen([z, c, s], training=False).numpy()
    return denorm_img(imgs)

def save_grid(imgs, outpath, title=None):
    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(imgs[i])
        ax.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default=r"C:\Users\calza\Poemario\datasetimg",
                    help="Ruta a datasetimg para leer conceptos/subconceptos")
    ap.add_argument("--list", action="store_true", help="Lista conceptos/subconceptos del datasetimg y sale")

    ap.add_argument("--concept", default="", help="Concepto (nombre de carpeta). Si vacío y no --list, error.")
    ap.add_argument("--subconcept", default="", help="Subconcepto (subcarpeta). Si vacío, se elige uno.")

    ap.add_argument("--n", type=int, default=9, help="Cuántas imágenes generar")
    ap.add_argument("--seed", type=int, default=1234, help="Seed base")
    ap.add_argument("--psi", type=float, default=0.7, help="Truncation (1.0=normal, <1 más consistente)")

    ap.add_argument("--save_path", default=DEFAULT_SAVE_PATH,
                    help="Carpeta donde guardaste el modelo/encoders")
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Nombre del .keras del Generator")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Carpeta de salida")
    ap.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)

    ap.add_argument("--fixed_seeds_json", default="", help="JSON para seeds fijas por etiqueta (opcional)")
    ap.add_argument("--use_fixed_seed", action="store_true",
                    help="Usa/crea una seed fija por (concept, subconcept)")

    ap.add_argument("--random_sub", action="store_true",
                    help="Si no pasas --subconcept, elige uno aleatorio del dataset")
    args = ap.parse_args()

    catalog = scan_dataset_structure(args.dataset_path)

    if args.list:
        print_catalog(catalog)
        return

    if not args.concept:
        raise ValueError("Debes pasar --concept o usar --list")

    if args.concept not in catalog:
        example = ", ".join(list(catalog.keys())[:40])
        raise ValueError(f"Concepto '{args.concept}' no existe en dataset. Ejemplos: {example} ...")

    subs = catalog[args.concept]
    subconcept = args.subconcept.strip()

    if not subconcept:
        subconcept = np.random.choice(subs) if args.random_sub else subs[0]

    if subconcept not in subs:
        example = ", ".join(subs[:40])
        raise ValueError(
            f"Subconcepto '{subconcept}' no existe dentro de '{args.concept}'. Ejemplos: {example} ..."
        )

    ensure_dir(args.outdir)

    gen, le_con, le_sub = load_assets(args.save_path, args.model)

    # Doble validación: debe existir en encoder también
    con_idx = label_to_idx(le_con, args.concept, kind="Concepto")
    sub_idx = label_to_idx(le_sub, subconcept, kind="Subconcepto")

    # Seeds fijas por etiqueta (opcional)
    seed_map = get_fixed_seed_map(args.fixed_seeds_json)
    key = make_key(args.concept, subconcept)

    seed_to_use = args.seed
    if args.use_fixed_seed:
        if key not in seed_map:
            seed_map[key] = int(np.random.randint(0, 2**31 - 1))
            if args.fixed_seeds_json:
                save_fixed_seed_map(args.fixed_seeds_json, seed_map)
        seed_to_use = seed_map[key]

    imgs = generate_images(
        gen=gen,
        con_idx=con_idx,
        sub_idx=sub_idx,
        n=args.n,
        latent_dim=args.latent_dim,
        seed=seed_to_use,
        psi=args.psi
    )

    safe_con = args.concept.replace(" ", "_")
    safe_sub = subconcept.replace(" ", "_")

    out_grid = os.path.join(args.outdir, f"{safe_con}__{safe_sub}__seed{seed_to_use}__psi{args.psi}.png")
    title = f"Concepto: {args.concept} | Sub: {subconcept} | seed={seed_to_use} | psi={args.psi}"
    save_grid(imgs, out_grid, title=title)

    indiv_dir = os.path.join(args.outdir, f"{safe_con}__{safe_sub}__seed{seed_to_use}__psi{args.psi}")
    ensure_dir(indiv_dir)
    for i, im in enumerate(imgs):
        plt.imsave(os.path.join(indiv_dir, f"img_{i:03d}.png"), im)

    print("✅ Listo.")
    print("📌 Grid:", out_grid)
    print("📌 Individuales:", indiv_dir)
    print("📌 Concept idx:", con_idx, "| Sub idx:", sub_idx)
    if args.use_fixed_seed:
        print("🔒 Seed fija por etiqueta:", seed_to_use)
        if args.fixed_seeds_json:
            print("🧾 JSON seeds:", args.fixed_seeds_json)

if __name__ == "__main__":
    main()
