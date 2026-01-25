import os
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
GEN_PATH = r"C:\Users\calza\Poemario\saved_models_final\gen_final.keras"
ENC_CON = r"C:\Users\calza\Poemario\saved_models_final\encoder_con.pkl"
ENC_SUB = r"C:\Users\calza\Poemario\saved_models_final\encoder_sub.pkl"

# Cargar recursos
gen = tf.keras.models.load_model(GEN_PATH, compile=False)
le_con = joblib.load(ENC_CON)
le_sub = joblib.load(ENC_SUB)

def generar_test(concepto, n=4):
    print(f"🎨 Generando {n} imágenes para: {concepto}...")
    try:
        con_idx = le_con.transform([concepto.lower()])[0]
    except:
        print(f"❌ Error: El concepto '{concepto}' no existe en el dataset.")
        print("Opciones:", list(le_con.classes_)[:10], "...")
        return

    # Ruido y etiquetas
    z = np.random.normal(0, 1, (n, 128)).astype("float32")
    c = np.full((n, 1), con_idx, dtype="int32")
    s = np.zeros((n, 1), dtype="int32") # Subconcepto por defecto

    # Predecir
    imgs = gen.predict([z, c, s])
    imgs = (imgs + 1.0) * 0.5
    imgs = np.clip(imgs, 0, 1)

    # Mostrar cuadrícula
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for i in range(n):
        axes[i].imshow(imgs[i])
        axes[i].axis("off")
    plt.suptitle(f"Concepto GAN: {concepto.upper()}")
    plt.show()

if __name__ == "__main__":
    print("Conceptos listos:", list(le_con.classes_))
    tema = input("Escribe el nombre de una carpeta (concepto): ")
    generar_test(tema)