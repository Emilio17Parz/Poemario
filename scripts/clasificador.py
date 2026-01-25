import os
import json
import re
import unicodedata
from pathlib import Path
from typing import Tuple, Dict, Any, List

# =========================
# CONFIG
# =========================
RUTAS_A_BUSCAR = [
    r"C:\Users\calza\Poemario",
    r"C:\Users\calza\Poemariov2",
]

OUTPUT_BASE = "./dataset_final_validado"
OUTPUT_REPORT = "./reporte_validacion.json"

CATEGORIAS_OFICIALES = [
    "soneto", "haiku", "tanka", "limerick", "oda", "elegia", "egloga", "epigrama",
    "romance", "decima_espinela", "redondilla", "cuarteta", "cuarteto", "serventesio",
    "terceto", "terceto_encadenado", "pareado", "silva", "copla", "seguidilla",
    "estrofa_safica", "estrofa_alcaica", "estancia", "balada", "villanelle",
    "sestina", "pantoum", "rondo", "rondeau", "triolet", "madrigal", "zejel",
    "moaxaja", "gacela", "cancion_petrarquista", "himno", "poema_en_prosa",
    "verso_libre", "versiculo", "acrostico", "palindromo_poetico", "poema_concreto",
    "poema_narrativo", "poema_dramatico", "poema_lirico", "poema_elegiaco",
    "poema_epico", "poema_satirico", "poema_didactico",
]

# =========================
# NORMALIZACIÓN / TEXTO
# =========================
def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip().replace(" ", "_")
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    return re.sub(r"[^a-z0-9_]", "", text)

def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")

def clean_text_basic(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()

def get_lines(texto: str) -> List[str]:
    texto = clean_text_basic(texto)
    return [l.strip() for l in texto.split("\n") if l.strip()]

def last_word(line: str) -> str:
    line = strip_accents(line.lower())
    line = re.sub(r"[^\w\s]", "", line)
    parts = line.split()
    return parts[-1] if parts else ""

# =========================
# SILABEADOR APROX (ES)
# =========================
VOWELS = set("aeiouy")
def approx_syllables_es(line: str) -> int:
    """
    Conteo aproximado: grupos de vocales en español.
    NO aplica sinalefa/diéresis exacta. Es heurístico.
    """
    s = strip_accents(line.lower())
    s = re.sub(r"[^a-zñáéíóúü\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return 0

    count = 0
    prev_vowel = False
    for ch in s:
        is_v = (ch in VOWELS) or (ch in "áéíóúü")
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    return max(count, 1)

def approx_meter_ok(line: str, target: int, tol: int) -> bool:
    n = approx_syllables_es(line)
    return (target - tol) <= n <= (target + tol)

# =========================
# RIMA APROX
# =========================
def rhyme_key_consonant(word: str, n: int = 4) -> str:
    w = strip_accents(word.lower())
    w = re.sub(r"[^a-z0-9ñ]", "", w)
    return w[-n:] if len(w) >= n else w

def rhyme_key_assonant(word: str) -> str:
    w = strip_accents(word.lower())
    w = re.sub(r"[^a-z0-9ñ]", "", w)
    vowels = "".join([c for c in w if c in "aeiou"])
    return vowels[-2:] if len(vowels) >= 2 else vowels

def build_scheme(keys: List[str]) -> str:
    """
    Dada una lista de 'claves de rima' por línea, genera un esquema tipo ABBA.
    """
    mapping = {}
    nxt = ord("A")
    scheme = []
    for k in keys:
        if k not in mapping:
            mapping[k] = chr(nxt)
            nxt += 1
        scheme.append(mapping[k])
    return "".join(scheme)

# =========================
# VALIDACIONES POR FORMA
# =========================
def validate_soneto(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) != 14:
        errs.append("Soneto: debe tener 14 versos.")
        return False, errs
    # Heurística métrica: endecasílabo ~11 (+/-1)
    bad = [i for i,l in enumerate(lines,1) if not approx_meter_ok(l, 11, tol=1)]
    if len(bad) >= 6:
        errs.append(f"Soneto: muchos versos no parecen endecasílabos (aprox). Líneas: {bad[:12]}")
    # Rima consonante aproximada: cuartetos suelen rimar, al menos debe haber repetición.
    keys = [rhyme_key_consonant(last_word(l), 4) for l in lines]
    scheme = build_scheme(keys)
    if len(set(scheme[:8])) > 4:  # muy disperso
        errs.append("Soneto: rima consonante poco consistente (heurística).")
    return (len(errs) == 0), errs

def validate_haiku(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) != 3:
        return False, ["Haiku: debe tener 3 versos."]
    targets = [5,7,5]
    for i,(l,t) in enumerate(zip(lines, targets), 1):
        if not approx_meter_ok(l, t, tol=1):
            errs.append(f"Haiku: línea {i} no parece {t} sílabas (aprox).")
    return (len(errs) == 0), errs

def validate_tanka(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) != 5:
        return False, ["Tanka: debe tener 5 versos."]
    targets = [5,7,5,7,7]
    for i,(l,t) in enumerate(zip(lines, targets), 1):
        if not approx_meter_ok(l, t, tol=1):
            errs.append(f"Tanka: línea {i} no parece {t} sílabas (aprox).")
    return (len(errs) == 0), errs

def validate_limerick(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) != 5:
        return False, ["Limerick: debe tener 5 versos."]
    # Rima AABBA (consonante aproximada)
    keys = [rhyme_key_consonant(last_word(l), 4) for l in lines]
    sch = build_scheme(keys)
    if sch not in ("AABBA", "AABBA"):  # esquema directo
        # AABBA puede mapearse a ABBA? no; mejor comparar por igualdad de claves:
        if not (keys[0]==keys[1]==keys[4] and keys[2]==keys[3] and keys[0]!=keys[2]):
            errs.append("Limerick: no cumple rima AABBA (aprox).")
    return (len(errs) == 0), errs

def validate_decima_espinela(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) != 10:
        return False, ["Décima espinela: debe tener 10 versos."]
    # Octosílabo ~8 (+/-1)
    bad = [i for i,l in enumerate(lines,1) if not approx_meter_ok(l, 8, tol=1)]
    if len(bad) >= 5:
        errs.append(f"Décima: muchos versos no parecen octosílabos (aprox). Líneas: {bad[:12]}")
    # Rima ABBAACCDDC (consonante aprox)
    keys = [rhyme_key_consonant(last_word(l), 4) for l in lines]
    pat = "ABBAACCDDC"
    # Convertimos a letras según primer aparición y comparamos patrón
    sch = build_scheme(keys)
    if sch != pat:
        errs.append(f"Décima: esquema de rima esperado {pat}, obtenido {sch} (aprox).")
    return (len(errs) == 0), errs

def validate_redondilla(lines: List[str]) -> Tuple[bool, List[str]]:
    if len(lines) != 4:
        return False, ["Redondilla: debe tener 4 versos."]
    errs = []
    # Octosílabo aprox
    bad = [i for i,l in enumerate(lines,1) if not approx_meter_ok(l, 8, tol=1)]
    if len(bad) >= 3:
        errs.append("Redondilla: métrica octosílaba poco consistente (aprox).")
    # ABBA consonante aprox
    keys = [rhyme_key_consonant(last_word(l), 4) for l in lines]
    if not (keys[0]==keys[3] and keys[1]==keys[2] and keys[0]!=keys[1]):
        errs.append("Redondilla: rima ABBA no detectada (aprox).")
    return (len(errs) == 0), errs

def validate_cuarteta(lines: List[str]) -> Tuple[bool, List[str]]:
    if len(lines) != 4:
        return False, ["Cuarteta: debe tener 4 versos."]
    errs = []
    bad = [i for i,l in enumerate(lines,1) if not approx_meter_ok(l, 8, tol=1)]
    if len(bad) >= 3:
        errs.append("Cuarteta: métrica octosílaba poco consistente (aprox).")
    keys = [rhyme_key_consonant(last_word(l), 4) for l in lines]
    if not (keys[0]==keys[2] and keys[1]==keys[3] and keys[0]!=keys[1]):
        errs.append("Cuarteta: rima ABAB no detectada (aprox).")
    return (len(errs) == 0), errs

def validate_cuarteto(lines: List[str]) -> Tuple[bool, List[str]]:
    if len(lines) != 4:
        return False, ["Cuarteto: debe tener 4 versos."]
    errs = []
    bad = [i for i,l in enumerate(lines,1) if not approx_meter_ok(l, 11, tol=1)]
    if len(bad) >= 3:
        errs.append("Cuarteto: endecasílabos poco consistentes (aprox).")
    keys = [rhyme_key_consonant(last_word(l), 4) for l in lines]
    ok_abab = (keys[0]==keys[2] and keys[1]==keys[3] and keys[0]!=keys[1])
    ok_abba = (keys[0]==keys[3] and keys[1]==keys[2] and keys[0]!=keys[1])
    if not (ok_abab or ok_abba):
        errs.append("Cuarteto: rima ABAB o ABBA no detectada (aprox).")
    return (len(errs) == 0), errs

def validate_serventesio(lines: List[str]) -> Tuple[bool, List[str]]:
    if len(lines) != 4:
        return False, ["Serventesio: debe tener 4 versos."]
    errs = []
    bad = [i for i,l in enumerate(lines,1) if not approx_meter_ok(l, 11, tol=1)]
    if len(bad) >= 3:
        errs.append("Serventesio: endecasílabos poco consistentes (aprox).")
    keys = [rhyme_key_consonant(last_word(l), 4) for l in lines]
    if not (keys[0]==keys[2] and keys[1]==keys[3] and keys[0]!=keys[1]):
        errs.append("Serventesio: rima ABAB no detectada (aprox).")
    return (len(errs) == 0), errs

def validate_terceto(lines: List[str]) -> Tuple[bool, List[str]]:
    if len(lines) != 3:
        return False, ["Terceto: debe tener 3 versos."]
    errs = []
    bad = [i for i,l in enumerate(lines,1) if not approx_meter_ok(l, 11, tol=1)]
    if len(bad) >= 2:
        errs.append("Terceto: endecasílabos poco consistentes (aprox).")
    # Rima variable: no obligamos patrón, solo que haya alguna rima
    keys = [rhyme_key_consonant(last_word(l), 4) for l in lines]
    if len(set(keys)) == 3:
        errs.append("Terceto: no se detecta rima (aprox).")
    return (len(errs) == 0), errs

def validate_pareado(lines: List[str]) -> Tuple[bool, List[str]]:
    if len(lines) != 2:
        return False, ["Pareado: debe tener 2 versos."]
    k1 = rhyme_key_consonant(last_word(lines[0]), 4)
    k2 = rhyme_key_consonant(last_word(lines[1]), 4)
    if k1 and k2 and k1 != k2:
        return False, ["Pareado: no se detecta rima consonante entre los dos versos (aprox)."]
    return True, []

def validate_romance(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) < 8:
        errs.append("Romance: suele tener serie indefinida; mínimo recomendado 8 versos.")
    # Octosílabo aprox
    bad = [i for i,l in enumerate(lines,1) if not approx_meter_ok(l, 8, tol=1)]
    if len(bad) > len(lines) * 0.6:
        errs.append("Romance: muchos versos no parecen octosílabos (aprox).")
    # Asonante en pares
    pares = [rhyme_key_assonant(last_word(lines[i])) for i in range(1, len(lines), 2)]
    if len(pares) >= 3 and len(set([p for p in pares if p])) > 2:
        errs.append("Romance: rima asonante en versos pares poco consistente (aprox).")
    return (len(errs) == 0), errs

def validate_seguidilla(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) != 4:
        errs.append("Seguidilla: suele ser 4 versos (7-5-7-5).")
        return False, errs
    targets = [7,5,7,5]
    for i,(l,t) in enumerate(zip(lines, targets), 1):
        if not approx_meter_ok(l, t, tol=1):
            errs.append(f"Seguidilla: línea {i} no parece {t} sílabas (aprox).")
    # Asonante en pares
    k2 = rhyme_key_assonant(last_word(lines[1]))
    k4 = rhyme_key_assonant(last_word(lines[3]))
    if k2 and k4 and k2 != k4:
        errs.append("Seguidilla: rima asonante en versos pares no detectada (aprox).")
    return (len(errs) == 0), errs

def validate_estrofa_safica(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) != 4:
        return False, ["Estrofa sáfica: 4 versos (3 endecasílabos + 1 pentasílabo)."]
    for i in range(3):
        if not approx_meter_ok(lines[i], 11, tol=1):
            errs.append(f"Estrofa sáfica: línea {i+1} no parece endecasílaba (aprox).")
    if not approx_meter_ok(lines[3], 5, tol=1):
        errs.append("Estrofa sáfica: línea 4 no parece pentasílaba (aprox).")
    return (len(errs) == 0), errs

def validate_villanelle(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) != 19:
        return False, ["Villanelle: debe tener 19 versos."]
    # Repetición de versos 1 y 3 (aprox exacto textual normalizado)
    def norm_line(s): return re.sub(r"\s+", " ", strip_accents(s.lower()).strip())
    l1 = norm_line(lines[0])
    l3 = norm_line(lines[2])
    must_be_l1 = [6, 12, 18]  # líneas 7,13,19 (0-indexed)
    must_be_l3 = [8, 14, 18]  # líneas 9,15,19
    for idx in must_be_l1:
        if idx < len(lines) and norm_line(lines[idx]) != l1:
            errs.append(f"Villanelle: línea {idx+1} debe repetir el verso 1.")
            break
    for idx in must_be_l3:
        if idx < len(lines) and norm_line(lines[idx]) != l3:
            errs.append(f"Villanelle: línea {idx+1} debe repetir el verso 3.")
            break
    # Dos rimas (muy heurístico)
    keys = [rhyme_key_consonant(last_word(l), 4) for l in lines if last_word(l)]
    if keys and len(set(keys)) > 4:
        errs.append("Villanelle: parecen existir más de dos rimas (heurística).")
    return (len(errs) == 0), errs

def validate_triolet(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) != 8:
        return False, ["Triolet: debe tener 8 versos."]
    def norm_line(s): return re.sub(r"\s+", " ", strip_accents(s.lower()).strip())
    l1 = norm_line(lines[0])
    l2 = norm_line(lines[1])
    if norm_line(lines[3]) != l1:
        errs.append("Triolet: línea 4 debe repetir la línea 1.")
    if norm_line(lines[6]) != l1:
        errs.append("Triolet: línea 7 debe repetir la línea 1.")
    if norm_line(lines[7]) != l2:
        errs.append("Triolet: línea 8 debe repetir la línea 2.")
    return (len(errs) == 0), errs

def validate_pantoum(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) < 8 or (len(lines) % 4 != 0):
        errs.append("Pantoum: debe tener estrofas de 4 versos (total múltiplo de 4) y al menos 2 estrofas.")
        return False, errs
    def norm_line(s): return re.sub(r"\s+", " ", strip_accents(s.lower()).strip())
    # Repetición: en cada estrofa, línea 2 y 4 pasan a ser 1 y 3 de la siguiente
    for s in range(0, len(lines) - 4, 4):
        a2 = norm_line(lines[s+1])
        a4 = norm_line(lines[s+3])
        b1 = norm_line(lines[s+4])
        b3 = norm_line(lines[s+6])
        if a2 != b1:
            errs.append(f"Pantoum: estrofa {s//4+2} línea 1 debe repetir estrofa {s//4+1} línea 2.")
            break
        if a4 != b3:
            errs.append(f"Pantoum: estrofa {s//4+2} línea 3 debe repetir estrofa {s//4+1} línea 4.")
            break
    return (len(errs) == 0), errs

def validate_sestina(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) not in (39, 40, 41):  # 6*6=36 + envío 3 (a veces 3/4)
        errs.append("Sestina: suele ser 6 estrofas de 6 versos (36) + envío final (~3).")
        return False, errs
    # Extraemos últimas palabras de los primeros 6 versos como "palabras finales"
    ends = [last_word(l) for l in lines[:6]]
    ends = [strip_accents(w.lower()) for w in ends if w]
    if len(ends) < 6 or len(set(ends)) < 5:
        errs.append("Sestina: no se detectan 6 palabras finales distintas en la primera estrofa.")
        return False, errs
    # Verificamos que en cada bloque de 6 versos, las palabras finales sean una permutación del set inicial
    base = set(ends[:6])
    for st in range(0, 36, 6):
        stanza = [strip_accents(last_word(l).lower()) for l in lines[st:st+6]]
        stanza = [w for w in stanza if w]
        if len(stanza) < 6 or set(stanza) != base:
            errs.append(f"Sestina: estrofa {st//6+1} no repite las 6 palabras finales (heurística).")
            break
    return (len(errs) == 0), errs

def validate_acrostico(lines: List[str], palabra: str) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) < 3:
        errs.append("Acróstico: muy corto (mínimo 3 versos recomendado).")
        return False, errs
    initials = "".join([strip_accents(l.strip()[0].lower()) for l in lines if l.strip()])
    initials = re.sub(r"[^a-z0-9ñ]", "", initials)
    target = strip_accents((palabra or "").lower())
    target = re.sub(r"[^a-z0-9ñ]", "", target)
    if target:
        if not initials.startswith(target):
            errs.append("Acróstico: las iniciales no forman (o no comienzan con) la palabra_clave_ingresada.")
            return False, errs
    else:
        # Sin palabra clave: al menos que forme algo "no trivial"
        if len(initials) < 4:
            errs.append("Acróstico: sin palabra clave y muy corto para validar.")
            return False, errs
    return True, []

def validate_palindromo_poetico(texto: str) -> Tuple[bool, List[str]]:
    s = strip_accents(texto.lower())
    s = re.sub(r"[^a-z0-9ñ]", "", s)
    if len(s) < 10:
        return False, ["Palíndromo poético: texto muy corto para validar."]
    if s != s[::-1]:
        return False, ["Palíndromo poético: no es palíndromo al normalizar."]
    return True, []

# Formas “flexibles” (no hay chequeo duro sin NLP avanzado)
def validate_flexible(lines: List[str], min_lines: int = 1, name: str = "Forma") -> Tuple[bool, List[str]]:
    if len(lines) < min_lines:
        return False, [f"{name}: texto muy corto."]
    return True, []

def validate_terceto_encadenado(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) < 7 or (len(lines) % 3 not in (0, 1, 2)):
        errs.append("Terceto encadenado: suele ser serie de tercetos (múltiplos de 3) + remate.")
        # no lo hacemos fatal si trae remate raro
    # Heurística de encadenado: rima ABA BCB CDC...
    keys = [rhyme_key_consonant(last_word(l), 4) for l in lines]
    # Validación suave: en cada terceto, línea 1 y 3 riman
    for i in range(0, len(lines) - 2, 3):
        k1, k3 = keys[i], keys[i+2]
        if k1 and k3 and k1 != k3:
            errs.append(f"Terceto encadenado: terceto {(i//3)+1} no cumple rima ABA (aprox).")
            break
    return (len(errs) == 0), errs

def validate_gacela(lines: List[str]) -> Tuple[bool, List[str]]:
    errs = []
    if len(lines) < 6 or len(lines) % 2 != 0:
        errs.append("Gacela (ghazal): suele ser pareados (número par de versos) y al menos 3 pareados.")
        return False, errs
    # Rima AA BA CA... -> detectamos que línea 2 de cada pareado comparte rima con línea 2 del primero
    k_ref = rhyme_key_consonant(last_word(lines[1]), 4)
    if not k_ref:
        errs.append("Gacela: no se detecta palabra final para rima.")
        return False, errs
    for i in range(3, len(lines), 2):
        ki = rhyme_key_consonant(last_word(lines[i]), 4)
        if ki and ki != k_ref:
            errs.append("Gacela: los segundos versos de cada pareado no comparten la rima (aprox).")
            break
    # Primer pareado AA (ambos versos comparten rima)
    k1 = rhyme_key_consonant(last_word(lines[0]), 4)
    if k1 and k_ref and k1 != k_ref:
        errs.append("Gacela: primer pareado no parece AA (aprox).")
    return (len(errs) == 0), errs

# Dispatcher: VALIDACIÓN PARA LAS 49
def validar_estructura(tipo: str, texto: str, palabra_clave: str = "") -> Tuple[bool, str, List[str]]:
    t = normalize(tipo)
    lines = get_lines(texto)

    if t not in CATEGORIAS_OFICIALES:
        return False, "complementaria", [f"Tipo '{tipo}' no está en las 49 oficiales tras normalizar ({t})."]

    # Validaciones específicas (duras o semi-duras)
    if t == "soneto":
        ok, errs = validate_soneto(lines)
    elif t == "haiku":
        ok, errs = validate_haiku(lines)
    elif t == "tanka":
        ok, errs = validate_tanka(lines)
    elif t == "limerick":
        ok, errs = validate_limerick(lines)
    elif t == "romance":
        ok, errs = validate_romance(lines)
    elif t == "decima_espinela":
        ok, errs = validate_decima_espinela(lines)
    elif t == "redondilla":
        ok, errs = validate_redondilla(lines)
    elif t == "cuarteta":
        ok, errs = validate_cuarteta(lines)
    elif t == "cuarteto":
        ok, errs = validate_cuarteto(lines)
    elif t == "serventesio":
        ok, errs = validate_serventesio(lines)
    elif t == "terceto":
        ok, errs = validate_terceto(lines)
    elif t == "terceto_encadenado":
        ok, errs = validate_terceto_encadenado(lines)
    elif t == "pareado":
        ok, errs = validate_pareado(lines)
    elif t == "seguidilla":
        ok, errs = validate_seguidilla(lines)
    elif t == "estrofa_safica":
        ok, errs = validate_estrofa_safica(lines)
    elif t == "villanelle":
        ok, errs = validate_villanelle(lines)
    elif t == "triolet":
        ok, errs = validate_triolet(lines)
    elif t == "pantoum":
        ok, errs = validate_pantoum(lines)
    elif t == "sestina":
        ok, errs = validate_sestina(lines)
    elif t == "acrostico":
        ok, errs = validate_acrostico(lines, palabra_clave)
    elif t == "palindromo_poetico":
        ok, errs = validate_palindromo_poetico(texto)
    elif t == "gacela":
        ok, errs = validate_gacela(lines)
    else:
        # Para el resto: validación mínima (no vacío) + algunas reglas suaves cuando aplica
        # (oda, elegia, egloga, epigrama, silva, copla, estancia, balada, rondo, rondeau, etc.)
        ok, errs = validate_flexible(lines, min_lines=1, name=t)

    return ok, (t if ok else "complementaria"), errs


# =========================
# “CORRECCIÓN 1”: NO SALTAR POR EXTRAS, LIMPIAR A SCHEMA
# =========================
def clean_to_schema(data: Any) -> Tuple[Dict[str, Any], List[str]]:
    """
    Devuelve un objeto que SOLO contiene:
      {
        "subcategoria": <str>,
        "poema": {
          "texto": <str>,
          "tipo": <str>,
          "palabra_clave_ingresada": <str>
        }
      }
    Si faltan cosas, las crea en blanco y deja warnings.
    """
    warnings = []
    cleaned = {"subcategoria": "", "poema": {"texto": "", "tipo": "", "palabra_clave_ingresada": ""}}

    if not isinstance(data, dict):
        warnings.append("JSON raíz no es dict; se normaliza a schema vacío.")
        return cleaned, warnings

    # subcategoria
    if "subcategoria" in data and isinstance(data["subcategoria"], str):
        cleaned["subcategoria"] = data["subcategoria"].strip()
    else:
        warnings.append("Falta 'subcategoria' o no es string; se dejó en blanco.")

    # poema
    p = data.get("poema", {})
    if not isinstance(p, dict):
        warnings.append("'poema' no es dict; se normaliza.")
        p = {}

    # texto
    txt = p.get("texto", "")
    if isinstance(txt, str):
        cleaned["poema"]["texto"] = clean_text_basic(txt)
    else:
        warnings.append("Falta 'poema.texto' o no es string; se dejó en blanco.")

    # tipo
    tip = p.get("tipo", "")
    if isinstance(tip, str) and tip.strip():
        cleaned["poema"]["tipo"] = tip.strip()
    else:
        warnings.append("Falta 'poema.tipo' o no es string; se dejó en blanco.")

    # palabra_clave_ingresada
    pci = p.get("palabra_clave_ingresada", "")
    if isinstance(pci, str):
        cleaned["poema"]["palabra_clave_ingresada"] = pci.strip()
    else:
        warnings.append("Falta 'poema.palabra_clave_ingresada' o no es string; se dejó en blanco.")

    return cleaned, warnings


# =========================
# PROCESO
# =========================
from collections import Counter, defaultdict
import random
import time

def procesar():
    print(f"Iniciando desde: {os.getcwd()}")

    conteo_total = 0
    conteo_validos = 0
    conteo_complementaria = 0

    # ===== Reporte agregado (NO 1.1M entradas) =====
    by_final_category = Counter()          # cat_final -> count
    by_type_normalized = Counter()         # normalize(tipo) -> count
    by_type_raw = Counter()                # tipo raw -> count (top)
    by_error_code = Counter()              # error_code -> count
    by_error_msg = Counter()               # mensaje exacto -> count (top)
    by_final_and_error = Counter()         # (cat_final, error_code) -> count

    # Muestras de ejemplos (para inspección)
    SAMPLE_LIMIT_PER_ERROR = 200
    samples_by_error = defaultdict(list)   # error_code -> [{src, tipo, texto_preview, errs}]
    rng = random.Random(12345)

    def error_code_from_msg(msg: str) -> str:
        """
        Convierte mensajes a códigos estables (heurístico).
        """
        m = msg.lower()
        if "no está en las 49 oficiales" in m or "no esta en las 49 oficiales" in m:
            return "TYPE_NOT_IN_49"
        if "debe tener" in m and "versos" in m:
            return "LINE_COUNT_MISMATCH"
        if "no parece" in m and "silab" in m:
            return "METER_FAIL"
        if "rima" in m and ("no detectada" in m or "no cumple" in m or "poco consistente" in m):
            return "RHYME_FAIL"
        if "debe repetir" in m:
            return "REPETITION_FAIL"
        if "palindromo" in m and "no es palindromo" in m:
            return "PALINDROME_FAIL"
        if "texto muy corto" in m:
            return "TOO_SHORT"
        if "json invalido" in m or "no se pudo parsear json" in m:
            return "JSON_PARSE_FAIL"
        return "OTHER"

    t0 = time.time()

    for ruta_base in RUTAS_A_BUSCAR:
        if not os.path.exists(ruta_base):
            print(f"Ruta no encontrada: {ruta_base}")
            continue

        print(f"Escaneando carpetas 'dataset*' en: {ruta_base}")

        try:
            subcarpetas = [
                d for d in os.listdir(ruta_base)
                if os.path.isdir(os.path.join(ruta_base, d)) and d.lower().startswith("dataset")
            ]
        except Exception as e:
            print(f"Error accediendo a {ruta_base}: {e}")
            continue

        for sub in subcarpetas:
            ruta_dataset = os.path.join(ruta_base, sub)
            print(f"  Procesando: {sub}")

            for root, _, files in os.walk(ruta_dataset):
                for file in files:
                    if not file.endswith(".json"):
                        continue

                    full_path = os.path.join(root, file)
                    conteo_total += 1

                    warnings = []
                    errores = []

                    # 1) Leer JSON
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            raw_data = json.load(f)
                    except Exception as e:
                        errores.append(f"No se pudo parsear JSON: {type(e).__name__}: {e}")
                        cleaned = {"subcategoria": "", "poema": {"texto": "", "tipo": "", "palabra_clave_ingresada": ""}}
                        cleaned["poema"]["tipo"] = "complementaria"

                        cat_final = "complementaria"

                        # Guardar stub (no se salta)
                        dest_folder = os.path.join(OUTPUT_BASE, cat_final)
                        os.makedirs(dest_folder, exist_ok=True)
                        dest_path = os.path.join(dest_folder, file)
                        if os.path.exists(dest_path):
                            pref = "v2_" if "poemariov2" in ruta_base.lower() else "v1_"
                            dest_path = os.path.join(dest_folder, pref + file)
                        with open(dest_path, "w", encoding="utf-8") as f:
                            json.dump(cleaned, f, ensure_ascii=False, indent=2)

                        conteo_complementaria += 1
                        by_final_category[cat_final] += 1

                        # errores -> códigos
                        for em in errores:
                            code = error_code_from_msg(em)
                            by_error_code[code] += 1
                            by_error_msg[em] += 1
                            by_final_and_error[(cat_final, code)] += 1

                            # sample
                            if len(samples_by_error[code]) < SAMPLE_LIMIT_PER_ERROR and rng.random() < 0.02:
                                samples_by_error[code].append({
                                    "src": full_path,
                                    "tipo": "",
                                    "texto_preview": "",
                                    "errores": errores[:5],
                                })
                        continue

                    # 2) limpiar a schema (quita extras)
                    cleaned, w = clean_to_schema(raw_data)
                    warnings.extend(w)

                    tipo_raw = cleaned["poema"]["tipo"]
                    texto = cleaned["poema"]["texto"]
                    palabra = cleaned["poema"]["palabra_clave_ingresada"]

                    tnorm = normalize(tipo_raw)
                    by_type_normalized[tnorm] += 1
                    if tipo_raw:
                        by_type_raw[tipo_raw] += 1

                    # 3) validar
                    es_valido, cat_final, errs_val = validar_estructura(tipo_raw, texto, palabra)
                    if errs_val:
                        errores.extend(errs_val)

                    # 4) guardar
                    dest_folder = os.path.join(OUTPUT_BASE, cat_final)
                    os.makedirs(dest_folder, exist_ok=True)

                    dest_path = os.path.join(dest_folder, file)
                    if os.path.exists(dest_path):
                        pref = "v2_" if "poemariov2" in ruta_base.lower() else "v1_"
                        dest_path = os.path.join(dest_folder, pref + file)

                    with open(dest_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned, f, ensure_ascii=False, indent=2)

                    by_final_category[cat_final] += 1

                    if cat_final != "complementaria":
                        conteo_validos += 1
                    else:
                        conteo_complementaria += 1

                    # errores -> códigos (solo si complementaria o si quieres siempre)
                    for em in errores:
                        code = error_code_from_msg(em)
                        by_error_code[code] += 1
                        by_error_msg[em] += 1
                        by_final_and_error[(cat_final, code)] += 1

                        # sample representativa (más agresivo para complementaria)
                        if cat_final == "complementaria":
                            if len(samples_by_error[code]) < SAMPLE_LIMIT_PER_ERROR and rng.random() < 0.01:
                                samples_by_error[code].append({
                                    "src": full_path,
                                    "tipo": tipo_raw,
                                    "texto_preview": "\n".join(get_lines(texto)[:6])[:800],
                                    "errores": errores[:5],
                                })

    dt = time.time() - t0

    # ======= Construir reporte final compacto =======
    reporte = {
        "procesados": conteo_total,
        "validos": conteo_validos,
        "complementaria": conteo_complementaria,
        "tiempo_seg": round(dt, 2),

        "conteo_por_categoria_final": dict(by_final_category.most_common()),
        "top_tipos_normalizados": dict(by_type_normalized.most_common(50)),
        "top_tipos_raw": dict(by_type_raw.most_common(50)),

        "conteo_por_error_code": dict(by_error_code.most_common()),
        "top_errores_texto": dict(by_error_msg.most_common(50)),

        # tabla de cruce: (categoria, error_code) -> count
        "cruce_categoria_error": {
            f"{k[0]}__{k[1]}": v for k, v in by_final_and_error.most_common(200)
        },

        # muestras para inspección manual
        "samples_by_error_code": samples_by_error,
    }

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_REPORT)), exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print("\n--- FINALIZADO ---")
    print(f"Total procesados: {conteo_total}")
    print(f"Válidos (oficiales y pasan reglas): {conteo_validos}")
    print(f"Complementaria: {conteo_complementaria}")
    print(f"Resultados en: {os.path.abspath(OUTPUT_BASE)}")
    print(f"Reporte (compacto) en: {os.path.abspath(OUTPUT_REPORT)}")


if __name__ == "__main__":
    procesar()
