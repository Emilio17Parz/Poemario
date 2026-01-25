# scripts/modeloimg/entrenar_gan.py
# Entrenador optimizado para imágenes condicionales por (concept, subconcept)
# - WSL2 + TF GPU (RTX) friendly
# - Dataset estable (decode_png/jpeg, shapes fijas, sampling aleatorio)
# - IMG_SIZE=256 (calidad mejor)
# - Discriminator con PROJECTION + Hinge Loss
# - Mixed precision + LossScaleOptimizer (más estable)
# - N_CRITIC pasos de D
# - Filtra paths faltantes (evita NOT_FOUND)
# - Guardado por época + samples consistentes + modelos finales
# - steps_per_epoch opcional (recomendado para épocas manejables)

from concurrent.futures import process
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import math
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import mixed_precision
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
# =========================================================
# 0) ESTABILIDAD TF
# =========================================================
# Evita algunos crasheos del remapper/meta_optimizer
tf.config.optimizer.set_jit(False)  # apaga XLA JIT


# Mixed precision (solo si hay GPU)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        mixed_precision.set_global_policy("mixed_float16")
    except RuntimeError as e:
        print(e)

# (Opcional) Desactiva XLA si te da mensajes raros de spawn
# tf.config.optimizer.set_jit(False)

# =========================================================
# 1) CONFIG
# =========================================================
# Rutas corregidas para Windows
DATASET_PATH = r"C:\Users\calza\Poemario\datasetimg"
SAVE_PATH = r"C:\Users\calza\Poemario\saved_models_final"
IMG_SIZE = 256
LATENT_DIM = 128
BATCH_SIZE = 8
EPOCHS = 100

SAMPLES_PER_CLASS = 800
N_CRITIC = 1
LR_G = 4e-4
LR_D = 1e-4

SAVE_EVERY = 5
SAMPLE_EVERY = 5
SAMPLE_N = 16
TRUNC_PSI = 0.7

# Para que las “épocas” sean manejables con dataset enorme:
# Si lo dejas None, Keras recorrerá "lo que dé" y verás steps enormes.
STEPS_PER_EPOCH = 5000  # ej: 5000

VALID_EXT = (".png", ".jpg", ".jpeg")

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, "samples"), exist_ok=True)

# =========================================================
# 2) PREPARACIÓN: lee concept/subconcept desde carpetas
# datasetimg/
#   Concept/
#     Subconcept/
# =========================================================

def prepare_data():
    print(" Escaneando dataset...")
    data = []

    # Iniciamos el escaneo de carpetas
    for root, dirs, files in os.walk(DATASET_PATH):
        # Filtramos solo archivos con extensiones válidas
        imgs = [f for f in files if f.lower().endswith(VALID_EXT)]
        if not imgs:
            continue

        # Extraemos concepto y subconcepto de la estructura de carpetas
        rel_path = os.path.relpath(root, DATASET_PATH)
        parts = rel_path.split(os.sep)

        if not parts or parts[0] in (".", ""):
            continue

        concept = parts[0]
        subconcept = parts[1] if len(parts) > 1 else concept

        # Mezclamos y tomamos una muestra
        random.shuffle(imgs)
        sampled_imgs = imgs[:SAMPLES_PER_CLASS]

        # VALIDACIÓN DE ARCHIVOS (Aquí estaba el error anterior)
        for img_name in sampled_imgs:
            full_path = os.path.join(root, img_name)
            try:
                # Solo agregamos si el archivo existe y es mayor a 1KB
                if os.path.exists(full_path) and os.path.getsize(full_path) > 1024:
                    data.append([full_path, concept, subconcept])
            except OSError:
                continue 

    if not data:
        raise RuntimeError(f"No se encontraron imágenes válidas en: {DATASET_PATH}")

    df = pd.DataFrame(data, columns=["path", "concept", "subconcept"])

    # Codificación de etiquetas
    le_con, le_sub = LabelEncoder(), LabelEncoder()
    df["con_idx"] = le_con.fit_transform(df["concept"])
    df["sub_idx"] = le_sub.fit_transform(df["subconcept"])

    # Guardar encoders para el generador futuro
    joblib.dump(le_con, os.path.join(SAVE_PATH, "encoder_con.pkl"))
    joblib.dump(le_sub, os.path.join(SAVE_PATH, "encoder_sub.pkl"))

    n_con = len(le_con.classes_)
    n_sub = len(le_sub.classes_)

    # Catálogo para la interfaz de generación
    catalog = df.groupby("concept")["subconcept"].apply(lambda x: sorted(list(set(x)))).to_dict()
    joblib.dump(catalog, os.path.join(SAVE_PATH, "catalog_concept_to_sub.pkl"))

    print(f" Total imágenes validadas: {len(df)}")
    print(f" Conceptos: {n_con} | Subconceptos: {n_sub}")
    return df, n_con, n_sub

# =========================================================
# 3) tf.data (Versión robusta con py_function)
# =========================================================
def get_dataset(df):
    paths = df["path"].values.astype(str)
    con_idx = df["con_idx"].values.astype(np.int32)
    sub_idx = df["sub_idx"].values.astype(np.int32)

    def load_and_decode_py(path):
        try:
            # Convertimos el tensor de entrada a string de Python
            p_str = path.numpy().decode('utf-8') 
            with open(p_str, 'rb') as f:
                img_str = f.read()
            img = tf.image.decode_image(img_str, channels=3, expand_animations=False)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], antialias=True)
            img = tf.cast(img, tf.float32)
            img = (img - 127.5) / 127.5
            return img
        except Exception:
            # Si falla, imagen negra de respaldo
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    def process_step(path, c, s):
        # Cargamos la imagen mediante la función de Python
        img = tf.py_function(load_and_decode_py, [path], tf.float32)
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        
        # Mantenemos c y s como tensores de 1 solo elemento
        c = tf.cast(tf.reshape(c, [1]), tf.int32)
        s = tf.cast(tf.reshape(s, [1]), tf.int32)
        
        return img, (c, s)

    # Creamos el dataset
    ds = tf.data.Dataset.from_tensor_slices((paths, con_idx, sub_idx))
    ds = ds.repeat() # Para que no se detenga al terminar las fotos
    ds = ds.shuffle(4096)
    
    # Mapeamos el proceso
    ds = ds.map(process_step, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Agrupamos en Batch (esto es lo que causaba el error de broadcast)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# =========================================================
# 4) MODELOS
# =========================================================
def g_block(x, f):
    x = layers.Conv2DTranspose(f, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def build_generator(n_con, n_sub):
    z = layers.Input(shape=(LATENT_DIM,), name="z")
    c = layers.Input(shape=(1,), dtype="int32", name="concept")
    s = layers.Input(shape=(1,), dtype="int32", name="subconcept")

    emb_c = layers.Flatten()(layers.Embedding(n_con, 64)(c))
    emb_s = layers.Flatten()(layers.Embedding(n_sub, 64)(s))

    x = layers.Concatenate()([z, emb_c, emb_s])
    x = layers.Dense(8 * 8 * 512)(x)
    x = layers.Reshape((8, 8, 512))(x)

    # 8 -> 16 -> 32 -> 64 -> 128 -> 256
    x = g_block(x, 256)
    x = g_block(x, 128)
    x = g_block(x, 64)
    x = g_block(x, 32)
    x = g_block(x, 16)

    out = layers.Conv2D(3, 3, padding="same", activation="tanh", dtype="float32")(x)
    return Model([z, c, s], out, name="Generator")

def d_block(x, f):
    x = layers.Conv2D(f, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def build_projection_discriminator(n_con, n_sub):
    img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="img")
    c = layers.Input(shape=(1,), dtype="int32", name="concept")
    s = layers.Input(shape=(1,), dtype="int32", name="subconcept")

    # 256 -> 128 -> 64 -> 32 -> 16 -> 8
    x = d_block(img, 64)
    x = d_block(x, 128)
    x = d_block(x, 256)
    x = d_block(x, 256)
    x = d_block(x, 256)

    FEATURE_DIM = 256

    x = layers.GlobalAveragePooling2D()(x)
    feat = layers.Dense(FEATURE_DIM)(x)

    # Salidas float32 (estabilidad con mixed precision)
    logits = layers.Dense(1, use_bias=True, dtype="float32")(feat)

    emb_c = layers.Embedding(n_con, FEATURE_DIM)(c)  # (B,1,256)
    emb_s = layers.Embedding(n_sub, FEATURE_DIM)(s)  # (B,1,256)
    emb_c = layers.Reshape((FEATURE_DIM,))(emb_c)    # (B,256)
    emb_s = layers.Reshape((FEATURE_DIM,))(emb_s)    # (B,256)

    proj_c = layers.Dot(axes=1, dtype="float32")([feat, emb_c])  # (B,1)
    proj_s = layers.Dot(axes=1, dtype="float32")([feat, emb_s])  # (B,1)

    out = layers.Add(dtype="float32")([logits, proj_c, proj_s])
    return Model([img, c, s], out, name="DiscriminatorProj")

# =========================================================
# 5) HINGE LOSS
# =========================================================
@tf.function
def d_hinge_loss(d_real, d_fake):
    return tf.reduce_mean(tf.nn.relu(1.0 - d_real)) + tf.reduce_mean(tf.nn.relu(1.0 + d_fake))

@tf.function
def g_hinge_loss(d_fake):
    return -tf.reduce_mean(d_fake)

# =========================================================
# 6) TRAINER (mixed precision correcto)
# =========================================================
class HingeGAN(Model):
    def __init__(self, gen, disc):
        super().__init__()
        self.gen = gen
        self.disc = disc

    def compile(self, g_opt, d_opt, **kwargs):
        super().compile(**kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt

    def train_step(self, data):
        real_img, (c, s) = data
        bs = tf.shape(real_img)[0]

        d_loss_acc = 0.0

        # ---- Train D N_CRITIC veces ----
        for _ in range(N_CRITIC):
            z = tf.random.normal((bs, LATENT_DIM))
            with tf.GradientTape() as tape:
                fake_img = self.gen([z, c, s], training=True)
                d_real = self.disc([real_img, c, s], training=True)
                d_fake = self.disc([fake_img, c, s], training=True)
                d_loss = d_hinge_loss(d_real, d_fake)
                scaled_d_loss = self.d_opt.get_scaled_loss(d_loss)

            scaled_grads = tape.gradient(scaled_d_loss, self.disc.trainable_variables)
            grads = self.d_opt.get_unscaled_gradients(scaled_grads)
            self.d_opt.apply_gradients(zip(grads, self.disc.trainable_variables))
            d_loss_acc += d_loss

        d_loss_acc /= float(N_CRITIC)
        

        # ---- Train G ----
        z = tf.random.normal((bs, LATENT_DIM))
        with tf.GradientTape() as tape:
            gen_img = self.gen([z, c, s], training=True)
            d_fake = self.disc([gen_img, c, s], training=True)
            g_loss = g_hinge_loss(d_fake)
            scaled_g_loss = self.g_opt.get_scaled_loss(g_loss)

        scaled_grads = tape.gradient(scaled_g_loss, self.gen.trainable_variables)
        grads = self.g_opt.get_unscaled_gradients(scaled_grads)
        self.g_opt.apply_gradients(zip(grads, self.gen.trainable_variables))

        return {"g_loss": g_loss, "d_loss": d_loss_acc}

# =========================================================
# 7) CALLBACKS: samples + guardado por época
# =========================================================
def denorm_uint8(x):
    x = (x + 1.0) * 0.5
    x = tf.clip_by_value(x, 0.0, 1.0)
    return tf.cast(x * 255.0, tf.uint8)

def make_grid(imgs_uint8):
    n = imgs_uint8.shape[0]
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    h, w = imgs_uint8.shape[1], imgs_uint8.shape[2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for i in range(n):
        r = i // cols
        c = i % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = imgs_uint8[i]
    return grid

class SampleCallback(tf.keras.callbacks.Callback):
    def __init__(self, gen, df):
        super().__init__()
        self.gen_ref = gen

        # Samples consistentes: usamos el modo para (concept,subconcept)
        self.con_idx = int(df["con_idx"].mode().iloc[0])
        self.sub_idx = int(df["sub_idx"].mode().iloc[0])

        rng = np.random.default_rng(1234)
        z = rng.normal(0, 1, size=(SAMPLE_N, LATENT_DIM)).astype(np.float32)
        self.fixed_z = z * TRUNC_PSI

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % SAMPLE_EVERY != 0:
            return

        c = np.full((SAMPLE_N, 1), self.con_idx, dtype=np.int32)
        s = np.full((SAMPLE_N, 1), self.sub_idx, dtype=np.int32)

        imgs = self.gen_ref([self.fixed_z, c, s], training=False)
        imgs = denorm_uint8(imgs).numpy()

        grid = make_grid(imgs)
        out = os.path.join(SAVE_PATH, "samples", f"epoch_{epoch+1:03d}.png")
        tf.keras.utils.save_img(out, grid)
        print(f" Sample guardado: {out}")

class SaveGenEach(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % SAVE_EVERY == 0:
            gen_path = os.path.join(SAVE_PATH, f"gen_epoch_{epoch+1:03d}.keras")
            self.model.gen.save(gen_path)
            print(f" Guardado: {gen_path}")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("TensorFlow NO está usando GPU. Revisa instalación CUDA/WSL.")

    df, n_c, n_s = prepare_data()
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    print("✅ Imágenes existentes:", len(df))

    ds = get_dataset(df)

    gen = build_generator(n_c, n_s)
    gen.load_weights(r"C:\Users\calza\Poemario\saved_models_final\gen_epoch_100.keras")
    print(" Pesos de la época 10 cargados. Reintentando con nuevos parámetros...")
    disc = build_projection_discriminator(n_c, n_s)

    gan = HingeGAN(gen, disc)

    # Optimizadores + LossScale (mixed precision correcto)
    g_opt = tf.keras.optimizers.Adam(LR_G, beta_1=0.0, beta_2=0.9, clipnorm=1.0)
    d_opt = tf.keras.optimizers.Adam(LR_D, beta_1=0.0, beta_2=0.9, clipnorm=1.0)
    g_opt = mixed_precision.LossScaleOptimizer(g_opt)
    d_opt = mixed_precision.LossScaleOptimizer(d_opt)

    gan.compile(g_opt=g_opt, d_opt=d_opt, run_eagerly=False)


    print(" Entrenamiento iniciado...")

    fit_kwargs = dict(
        x=ds,
        epochs=EPOCHS,
        callbacks=[SampleCallback(gen, df), SaveGenEach()],
    )
    if STEPS_PER_EPOCH is not None:
        fit_kwargs["steps_per_epoch"] = STEPS_PER_EPOCH

    gan.fit(ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[SampleCallback(gen, df), SaveGenEach()])

    gen.save(os.path.join(SAVE_PATH, "gen_final.keras"))
    disc.save(os.path.join(SAVE_PATH, "disc_final.keras"))
    print(" Modelos guardados en:", SAVE_PATH)
