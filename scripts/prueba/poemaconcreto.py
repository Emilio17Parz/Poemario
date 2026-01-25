# ===== poema_concreto_generator_v3.py =====
import random
import re
from collections import defaultdict

OBJETOS = [
    "un ticket arrugado", "un vaso con hielo que cruje", "una ventana empañada",
    "un cable USB rebelde", "una llave que no gira", "una puerta que rechina",
    "un teclado pegajoso", "un cargador caliente", "una taza con café frío",
    "un plato astillado", "una bicicleta oxidada", "una caja de tornillos",
    "un trapo húmedo", "una cinta adhesiva gastada", "una moneda vieja",
]

# Palabras clave SOLO objetuales/sensoriales (evita abstractos)
KW = {
    "aereo": ["bruma", "polvo", "humo", "vaho", "olor"],
    "liquido": ["agua", "café", "aceite", "tinta", "condensación"],
    "superficie": ["grasa", "óxido", "salitre", "hollín", "arena", "pegamento"],
    "luz": ["reflejo", "sombra", "destello", "penumbra", "parpadeo"],
    "sonido": ["zumbido", "crujido", "clic", "chirrido", "estática", "susurro"],
    "solido": ["vidrio", "metal", "hierro", "cobre", "plástico", "cerámica", "tornillo"],
    "borde": ["grieta", "astilla", "raspón", "marca"],
}

# Verbos compatibles por clase (ojo: sonido separado con lógica propia)
VERBOS = {
    "aereo": ["flota", "se cuela", "se dispersa", "se queda suspendido"],
    "liquido": ["gotea", "se escurre", "moja", "deja mancha"],
    "superficie": ["se pega", "ensucia", "se incrusta", "deja rastro"],
    "luz": ["parpadea", "tiembla", "rebota", "se rompe", "se estira"],
    "solido": ["pesa", "se enfría", "se calienta", "tintinea", "golpea"],
    "borde": ["abre", "corta", "marca", "se nota", "se alarga"],
}

# Sonido: verbos por tipo (para evitar “zumbido cruje”)
VERBOS_SONIDO = {
    "zumbido": ["insiste", "vibra", "se mantiene", "tiembla"],
    "crujido": ["se abre", "responde", "revienta", "aparece"],
    "clic": ["salta", "corta", "marca", "se repite"],
    "chirrido": ["raspa", "sube", "se arrastra", "se clava"],
    "estática": ["chispea", "muerde", "se corta", "se derrama"],
    "susurro": ["se esconde", "roza", "se acerca", "se va"],
}

# Adjetivos por sentido (sin mezclar)
TACTO = ["áspero", "liso", "pegajoso", "frío", "tibio", "húmedo", "seco", "rugoso"]
VISTA = ["opaco", "brillante", "borroso", "nítido", "amarillento", "gris", "manchado"]
OLFATO = ["a humedad", "a metal", "a humo", "dulzón", "ácido"]
OIDO = ["bajo", "agudo", "grave", "intermitente", "seco", "constante"]

# Estructuras variadas (sin muletillas fijas)
ESTRUCTURAS = ["observacion", "inventario", "micro", "mecanismo", "contraste"]

def _pick_kw():
    clase = random.choice(list(KW.keys()))
    palabra = random.choice(KW[clase])
    return clase, palabra

def _linea_sentidos():
    return f"Al tacto: {random.choice(TACTO)}. A la vista: {random.choice(VISTA)}."

def _linea_olfato_o_sonido():
    if random.random() < 0.5:
        return f"Al olor: {random.choice(OLFATO)}."
    else:
        return f"Al oído: {random.choice(OIDO)}."

def poema_concreto():
    objeto = random.choice(OBJETOS)
    clase, palabra = _pick_kw()

    # acción principal coherente
    if clase == "sonido":
        verbo = random.choice(VERBOS_SONIDO.get(palabra, ["insiste"]))
        accion = f"{palabra} {verbo}."
    else:
        verbo = random.choice(VERBOS[clase])
        accion = f"{palabra} {verbo}."

    est = random.choice(ESTRUCTURAS)

    aperturas = [
        f"{objeto}.",
        f"Frente a {objeto}.",
        f"Hoy: {objeto}.",
        f"{objeto}, sin prisa.",
        f"{objeto}: lo que queda."
    ]
    cierres = [
        "No hay moraleja: queda el detalle.",
        "Lo cotidiano se mide por fricción.",
        "La materia no explica: insiste.",
        "Lo simple no promete; responde.",
        "Me llevo la escena, no la teoría."
    ]

    if est == "observacion":
        lines = [
            random.choice(aperturas),
            accion,
            _linea_sentidos(),
            _linea_olfato_o_sonido(),
            random.choice(cierres)
        ]
    elif est == "inventario":
        lines = [
            random.choice(aperturas),
            accion,
            "Inventario:",
            f"- una esquina {random.choice(['rayada','húmeda','gastada','templada','manchada'])}",
            f"- un sonido {random.choice(OIDO)}",
            f"- una luz {random.choice(['débil','oblicua','blanca','amarilla','corta'])}",
            random.choice(cierres)
        ]
    elif est == "micro":
        lines = [
            random.choice(aperturas),
            f"{accion[:-1]} cuando intento acomodarlo.",
            _linea_sentidos(),
            f"Se escucha: {random.choice(['un crujido','un clic','un zumbido','un chirrido','nada'])}.",
            "No pasa nada grande; pasa lo suficiente.",
            random.choice(cierres)
        ]
    elif est == "mecanismo":
        lines = [
            random.choice(aperturas),
            f"Primero {palabra}. Luego el resto.",
            f"{accion[:-1]} y deja evidencia.",
            f"Falla por {random.choice(['uso','humedad','cansancio','exceso','olvido'])}.",
            random.choice(cierres)
        ]
    else:  # contraste
        lines = [
            random.choice(aperturas),
            accion,
            f"Parece {random.choice(['normal','nuevo','quieto'])}, pero es {random.choice(['terco','frágil','limitado'])}.",
            _linea_olfato_o_sonido(),
            random.choice(cierres)
        ]

    texto = "\n".join(lines).strip()
    return palabra, texto

def repeticion_ngram(text: str, n=4) -> float:
    toks = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
    grams = [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
    if not grams:
        return 0.0
    freq = defaultdict(int)
    for g in grams: freq[g] += 1
    rep = sum(v-1 for v in freq.values() if v > 1)
    return rep / max(1, len(grams))
