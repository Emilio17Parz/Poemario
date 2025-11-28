from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "distiluse-base-multilingual-cased-v1"
model = SentenceTransformer(MODEL_NAME)

TEMAS = {
    "amor": ["beso", "pasión", "mirada", "corazón", "caricia"],
    "muerte": ["tumba", "oscuridad", "luto", "cementerio", "alma"],
    "naturaleza": ["árbol", "mariposa", "río", "montaña", "cielo"],
    "tristeza": ["llanto", "soledad", "ausencia", "dolor", "olvido"],
    "felicidad": ["risa", "luz", "alegría", "canto", "brillo"],
    "batalla": ["espada", "sangre", "reino", "honor", "guerra"]
}

def clasificar_frase(frase: str) -> dict:
    emb_frase = model.encode(frase.lower(), convert_to_tensor=True)

    mejor_tema = None
    mejor_sim = -1

    for tema, palabras in TEMAS.items():
        emb_palabras = model.encode([p.lower() for p in palabras], convert_to_tensor=True)
        sim = util.cos_sim(emb_frase, emb_palabras).mean().item()

        if sim > mejor_sim:
            mejor_sim = sim
            mejor_tema = tema

    return {
        "frase_original": frase,
        "tema_detectado": mejor_tema,
        "confianza": round(float(mejor_sim), 4)
    }


if __name__ == "__main__":
    frase = input("Ingresa una frase: ")
    print(clasificar_frase(frase))
