# -*- coding: utf-8 -*-
"""
Reclasifica y repara poemas desde ./complementaria hacia ./dataset_final_validado/<tipo>.

Uso:
  python .\scripts\reclasificaor.py --limit 60 --model gemma3 --report .\reporte_cambios.json
  python .\scripts\reclasificaor.py --dry-run --limit 20
  python .\scripts\reclasificaor.py --report-include-text   (incluye texto completo en el reporte)

Notas:
- Estrategia conservadora: normaliza + arreglos mínimos.
- Si no pasa validación, usa Ollama para reescribir SOLO lo necesario respetando reglas del tipo.
- Repara mojibake (Ã©, Ã±, etc.) antes de validar y antes de guardar.
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import ollama  # pip install ollama


# -------------------------
# CONFIG (paths)
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR_DEFAULT = ROOT / "complementaria/complementaria_failed"
DST_DIR_DEFAULT = ROOT / "dataset_final_validado"


# -------------------------
# TIPOS CANÓNICOS (carpetas destino)
# -------------------------
CANON: Dict[str, str] = {
    "acrostico": "acrostico",
    "balada": "balada",
    "cancion_petrarquista": "cancion_petrarquista",
    "copla": "copla",
    "cuarteta": "cuarteta",
    "cuarteto": "cuarteto",
    "decima_espinela": "decima_espinela",
    "egloga": "egloga",
    "elegia": "elegia",
    "epigrama": "epigrama",
    "estancia": "estancia",
    "estrofa_alcaica": "estrofa_alcaica",
    "estrofa_safica": "estrofa_safica",
    "gacela": "gacela",
    "haiku": "haiku",
    "himno": "himno",
    "limerick": "limerick",
    "madrigal": "madrigal",
    "moaxaja": "moaxaja",
    "oda": "oda",
    "palindromo_poetico": "palindromo_poetico",
    "pantoum": "pantoum",
    "pareado": "pareado",
    "poema_concreto": "poema_concreto",
    "poema_didactico": "poema_didactico",
    "poema_dramatico": "poema_dramatico",
    "poema_elegiaco": "poema_elegiaco",
    "poema_en_prosa": "poema_en_prosa",
    "poema_epico": "poema_epico",
    "poema_lirico": "poema_lirico",
    "poema_narrativo": "poema_narrativo",
    "poema_satirico": "poema_satirico",
    "redondilla": "redondilla",
    "romance": "romance",
    "rondeau": "rondeau",
    "rondo": "rondo",
    "seguidilla": "seguidilla",
    "serventesio": "serventesio",
    "sestina": "sestina",
    "silva": "silva",
    "soneto": "soneto",
    "tanka": "tanka",
    "terceto": "terceto",
    "terceto_encadenado": "terceto_encadenado",
    "triolet": "triolet",
    "versiculo": "versiculo",
    "verso_libre": "verso_libre",
    "zejel": "zejel",
    # carpeta de fallos
    "complementaria_failed": "complementaria_failed",
}

# Mapeos de nombres posibles -> canon
ALIASES: Dict[str, str] = {
    "acróstico": "acrostico",
    "acrostico": "acrostico",
    "cancion petrarquista": "cancion_petrarquista",
    "canción petrarquista": "cancion_petrarquista",
    "cancion_petrarquista": "cancion_petrarquista",
    "décima espinela": "decima_espinela",
    "decima espinela": "decima_espinela",
    "decima_espinela": "decima_espinela",
    "égloga": "egloga",
    "egloga": "egloga",
    "elegía": "elegia",
    "elegia": "elegia",
    "estrofa sáfica": "estrofa_safica",
    "estrofa safica": "estrofa_safica",
    "estrofa_safica": "estrofa_safica",
    "estrofa alcaica": "estrofa_alcaica",
    "estrofa_alcaica": "estrofa_alcaica",
    "palíndromo poético": "palindromo_poetico",
    "palindromo poetico": "palindromo_poetico",
    "palindromo_poetico": "palindromo_poetico",
    "poema en prosa": "poema_en_prosa",
    "poema_en_prosa": "poema_en_prosa",
    "poema didáctico": "poema_didactico",
    "poema didactico": "poema_didactico",
    "poema_didactico": "poema_didactico",
    "poema dramático": "poema_dramatico",
    "poema dramatico": "poema_dramatico",
    "poema_dramatico": "poema_dramatico",
    "poema elegíaco": "poema_elegiaco",
    "poema elegiaco": "poema_elegiaco",
    "poema_elegiaco": "poema_elegiaco",
    "poema épico": "poema_epico",
    "poema epico": "poema_epico",
    "poema_epico": "poema_epico",
    "poema lírico": "poema_lirico",
    "poema lirico": "poema_lirico",
    "poema_lirico": "poema_lirico",
    "poema narrativo": "poema_narrativo",
    "poema_narrativo": "poema_narrativo",
    "poema satírico": "poema_satirico",
    "poema satirico": "poema_satirico",
    "poema_satirico": "poema_satirico",
    "terceto encadenado": "terceto_encadenado",
    "terceto_encadenado": "terceto_encadenado",
    "versículo": "versiculo",
    "versiculo": "versiculo",
    "verso libre": "verso_libre",
    "verso_libre": "verso_libre",
    "zéjel": "zejel",
    "zejel": "zejel",
    # ya canon
    **{k: k for k in CANON if not k.endswith("_failed")},
}


# -------------------------
# REGLAS (prompt + validación básica)
# -------------------------
REGLAS: Dict[str, str] = {
    "haiku": "3 versos de 5, 7 y 5 sílabas aprox. Sin rima.",
    "tanka": "5 versos de 5-7-5-7-7 sílabas aprox.",
    "seguidilla": "4 versos con patrón 7-5-7-5 sílabas aprox.",
    "pareado": "2 versos que rimen entre sí (idealmente consonante).",
    "cuarteta": "4 versos de 8 sílabas aprox. Rima consonante cruzada (abab).",
    "redondilla": "4 versos de 8 sílabas aprox. Rima abrazada (abba).",
    "cuarteto": "4 versos de 11 sílabas aprox. Rima consonante abrazada (ABBA).",
    "romance": "Versos de 8 sílabas aprox. Rima asonante en los pares (aprox).",
    "soneto": "14 versos endecasílabos aprox (2 cuartetos + 2 tercetos).",
    "poema_en_prosa": "Prosa poética: principalmente un párrafo, pocos saltos, ritmo e imágenes.",
    "verso_libre": "Sin métrica ni rima fija, pero coherente y poético.",
    "versiculo": "Versos largos con ritmo solemne (estilo bíblico).",
}
REGLA_GENERIC = "Respeta el tipo indicado, en español, y produce un poema completo coherente."

# -------------------------
# DETECCIÓN: poema concreto / visual
# -------------------------
FORCE_ON_CONCRETE = True
CONCRETE_FALLBACK_TIPO = "verso_libre"  # o "poema_en_prosa" si prefieres

def detect_poema_concreto_visual(texto: str) -> bool:
    """
    Heurística para detectar poema concreto/visual:
    - secuencias largas de espacios/tabuladores
    - muchas líneas muy cortas (forma)
    - indentación muy variable
    - centrado/“escalones”
    """
    if not isinstance(texto, str) or not texto.strip():
        return False

    t = texto.replace("\r\n", "\n").replace("\r", "\n")

    if re.search(r"[ \t]{6,}", t):
        return True

    lines = [ln.rstrip("\n") for ln in t.split("\n") if ln.strip()]
    if len(lines) < 4:
        return False

    short = sum(1 for ln in lines if len(ln.strip()) <= 12)
    if short / max(1, len(lines)) >= 0.55:
        return True

    indents = [len(ln) - len(ln.lstrip(" \t")) for ln in lines]
    if len(set(indents)) >= 5 and (max(indents) - min(indents)) >= 8:
        return True

    centeredish = sum(1 for ln in lines if (len(ln) - len(ln.lstrip(" "))) >= 4 and len(ln.strip()) <= 30)
    if centeredish / max(1, len(lines)) >= 0.45:
        return True

    return False


# -------------------------
# MOJIBAKE
# -------------------------
def looks_mojibake(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    bad = ["Ã", "â€", "Â", "�"]
    return any(x in s for x in bad)

def fix_mojibake(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s or ""
    if not looks_mojibake(s):
        return s
    # latin1 -> utf8
    try:
        return s.encode("latin1", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        pass
    # cp1252 -> utf8
    try:
        return s.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return s


# -------------------------
# NORMALIZACIÓN
# -------------------------
def strip_accents(s: str) -> str:
    repl = {
        "á":"a","é":"e","í":"i","ó":"o","ú":"u","ü":"u","ñ":"n",
        "Á":"a","É":"e","Í":"i","Ó":"o","Ú":"u","Ü":"u","Ñ":"n",
    }
    return "".join(repl.get(ch, ch) for ch in (s or ""))

def norm_spaces(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def normalize_tipo(raw: str) -> str:
    raw = fix_mojibake(raw or "").strip().lower()
    raw = re.sub(r"[_\-]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = strip_accents(raw)

    if raw in ALIASES:
        return ALIASES[raw]
    raw2 = raw.replace(" ", "_")
    if raw2 in ALIASES:
        return ALIASES[raw2]
    if raw2 in CANON:
        return raw2
    return ""


# -------------------------
# JSON helpers
# -------------------------
def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        txt = path.read_text(encoding="utf-8", errors="strict")
        return json.loads(txt)
    except Exception:
        try:
            txt = path.read_text(encoding="latin1", errors="ignore")
            txt = fix_mojibake(txt)
            return json.loads(txt)
        except Exception:
            return None

def clean_to_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    subcat = fix_mojibake(str(obj.get("subcategoria", "") or ""))
    poema = obj.get("poema", {}) if isinstance(obj.get("poema", {}), dict) else {}
    texto = fix_mojibake(str(poema.get("texto", "") or ""))
    tipo = fix_mojibake(str(poema.get("tipo", "") or ""))
    key = fix_mojibake(str(poema.get("palabra_clave_ingresada", "") or ""))

    return {
        "subcategoria": subcat,
        "poema": {
            "texto": texto,
            "tipo": tipo,
            "palabra_clave_ingresada": key
        }
    }

def lines_from_text(texto: str) -> List[str]:
    texto = norm_spaces(texto)
    return [ln.strip() for ln in texto.split("\n") if ln.strip()]

def join_lines(lines: List[str]) -> str:
    return "\n".join([ln.strip() for ln in lines if ln.strip()]).strip()

def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# -------------------------
# MÉTRICA SIMPLE (aprox) para español
# -------------------------
VOWELS = set("aeiouáéíóúü")

def count_syllables_word(w: str) -> int:
    w = w.lower()
    w = re.sub(r"[^a-záéíóúüñ]", "", w)
    if not w:
        return 0
    syl = 0
    prev_v = False
    for ch in w:
        is_v = ch in VOWELS
        if is_v and not prev_v:
            syl += 1
        prev_v = is_v
    return max(1, syl)

def count_syllables_line(line: str) -> int:
    words = [w for w in re.split(r"\s+", line.strip()) if w]
    return sum(count_syllables_word(w) for w in words)

def approx_ok(actual: int, target: int, tol: int = 1) -> bool:
    return abs(actual - target) <= tol


# -------------------------
# RIMA / PAREADO
# -------------------------
def last_word(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    line = re.sub(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]+$", "", line)
    parts = re.split(r"\s+", line)
    return parts[-1] if parts else ""

def rhyme_key_consonant(word: str, n: int = 3) -> str:
    w = strip_accents((word or "").lower())
    w = re.sub(r"[^a-zñ]", "", w)
    return w[-n:] if len(w) >= n else w

def simple_rhyme_variants(word: str) -> List[str]:
    w = strip_accents((word or "").lower())
    w = re.sub(r"[^a-zñ]", "", w)
    if len(w) < 4:
        return []
    suf = w[-3:]
    if suf == "or":
        return ["amor", "dolor", "valor", "rumor"]
    if suf == "ar":
        return ["cantar", "soñar", "andar", "mirar"]
    if suf == "ion":
        return ["cancion", "pasion", "razon", "ilusion"]
    if suf == "ad":
        return ["verdad", "bondad", "piedad", "soledad"]
    if suf == "iel":
        return ["miel", "piel", "fiel"]
    if suf == "ono":
        return ["tono", "trono", "cono"]
    return []

def force_rhyme_without_dup(line2: str, target_word: str) -> str:
    tw = last_word(target_word) or ""
    tw_norm = strip_accents(tw.lower())
    if not tw_norm:
        return line2
    cand = simple_rhyme_variants(tw_norm)
    if not cand:
        return line2
    chosen = cand[0]
    parts = line2.split()
    if not parts:
        return line2
    parts[-1] = chosen
    return " ".join(parts)

def pareado_fix(lines: List[str]) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    if len(lines) < 2:
        return lines, warnings

    l1, l2 = lines[0], lines[1]
    w1 = last_word(l1)
    w2 = last_word(l2)

    k1 = rhyme_key_consonant(w1, 3)
    k2 = rhyme_key_consonant(w2, 3)

    if k1 and k2 and k1 == k2:
        return [l1, l2], warnings

    if strip_accents(w2.lower()) == strip_accents(w1.lower()):
        l2n = force_rhyme_without_dup(l2, w1)
        warnings.append("Pareado: evitó duplicar palabra final (variante simple).")
        return [l1, l2n], warnings

    l2n = force_rhyme_without_dup(l2, w1)
    if l2n != l2:
        warnings.append("Pareado: ajustó palabra final para rima (heurística).")
    return [l1, l2n], warnings


# -------------------------
# VALIDACIÓN BÁSICA POR TIPO (suave)
# -------------------------
def validate(tipo: str, texto: str) -> Dict[str, Any]:
    tipo = tipo or ""
    texto = norm_spaces(texto)
    lines = lines_from_text(texto)

    if not tipo:
        return {"ok": False, "categoria": "complementaria", "errores": ["Sin tipo."]}

    if tipo == "poema_en_prosa":
        # permitimos 1-3 líneas, pero no poema versificado largo
        if "\n" in texto and len(lines) > 3:
            return {"ok": False, "categoria": "complementaria", "errores": ["Poema en prosa: demasiados saltos de línea."]}
        if len(texto) < 80:
            return {"ok": False, "categoria": "complementaria", "errores": ["Poema en prosa: demasiado corto."]}
        return {"ok": True, "categoria": tipo, "errores": []}

    if tipo == "haiku":
        if len(lines) != 3:
            return {"ok": False, "categoria": "complementaria", "errores": ["Haiku: no tiene 3 versos."]}
        syl = [count_syllables_line(l) for l in lines]
        if not (approx_ok(syl[0], 5) and approx_ok(syl[1], 7) and approx_ok(syl[2], 5)):
            return {"ok": False, "categoria": "complementaria", "errores": [f"Haiku: sílabas aprox {syl} (esperado 5-7-5)."]}
        return {"ok": True, "categoria": tipo, "errores": []}

    if tipo == "tanka":
        if len(lines) != 5:
            return {"ok": False, "categoria": "complementaria", "errores": ["Tanka: no tiene 5 versos."]}
        syl = [count_syllables_line(l) for l in lines]
        target = [5, 7, 5, 7, 7]
        if not all(approx_ok(syl[i], target[i]) for i in range(5)):
            return {"ok": False, "categoria": "complementaria", "errores": [f"Tanka: sílabas aprox {syl} (esperado 5-7-5-7-7)."]}
        return {"ok": True, "categoria": tipo, "errores": []}

    if tipo == "seguidilla":
        if len(lines) != 4:
            return {"ok": False, "categoria": "complementaria", "errores": ["Seguidilla: no tiene 4 versos."]}
        syl = [count_syllables_line(l) for l in lines]
        target = [7, 5, 7, 5]
        if not all(approx_ok(syl[i], target[i]) for i in range(4)):
            return {"ok": False, "categoria": "complementaria", "errores": [f"Seguidilla: sílabas aprox {syl} (esperado 7-5-7-5)."]}
        return {"ok": True, "categoria": tipo, "errores": []}

    if tipo == "pareado":
        if len(lines) != 2:
            return {"ok": False, "categoria": "complementaria", "errores": ["Pareado: no tiene 2 versos."]}
        w1 = last_word(lines[0])
        w2 = last_word(lines[1])
        k1 = rhyme_key_consonant(w1, 3)
        k2 = rhyme_key_consonant(w2, 3)
        if not k1 or not k2 or k1 != k2:
            return {"ok": False, "categoria": "complementaria", "errores": ["Pareado: no riman (aprox)."]}
        return {"ok": True, "categoria": tipo, "errores": []}

    # default: mínimo
    if len(texto) < 40:
        return {"ok": False, "categoria": "complementaria", "errores": ["Poema demasiado corto."]}
    return {"ok": True, "categoria": tipo, "errores": []}


# -------------------------
# REPARACIÓN CONSERVADORA
# -------------------------
def conservative_repair(tipo: str, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    out = json.loads(json.dumps(schema, ensure_ascii=False))  # deep copy

    out["subcategoria"] = fix_mojibake((out.get("subcategoria") or "").strip())
    out["poema"]["texto"] = norm_spaces(fix_mojibake(out["poema"].get("texto", "")))
    out["poema"]["tipo"] = fix_mojibake(tipo)
    out["poema"]["palabra_clave_ingresada"] = fix_mojibake((out["poema"].get("palabra_clave_ingresada") or "").strip())

    # limpiezas extra
    out["poema"]["texto"] = re.sub(r"[ \t]+\n", "\n", out["poema"]["texto"])
    out["poema"]["texto"] = re.sub(r"\n[ \t]+", "\n", out["poema"]["texto"])

    if tipo == "pareado":
        lines = lines_from_text(out["poema"]["texto"])
        if len(lines) >= 2:
            lines = lines[:2]
            new_lines, w = pareado_fix(lines)
            warnings.extend(w)
            out["poema"]["texto"] = join_lines(new_lines)

    if tipo == "haiku":
        lines = lines_from_text(out["poema"]["texto"])
        if len(lines) > 3:
            out["poema"]["texto"] = join_lines(lines[:3])
            warnings.append("Haiku: recortó a 3 versos.")
    if tipo == "seguidilla":
        lines = lines_from_text(out["poema"]["texto"])
        if len(lines) > 4:
            out["poema"]["texto"] = join_lines(lines[:4])
            warnings.append("Seguidilla: recortó a 4 versos.")
    if tipo == "tanka":
        lines = lines_from_text(out["poema"]["texto"])
        if len(lines) > 5:
            out["poema"]["texto"] = join_lines(lines[:5])
            warnings.append("Tanka: recortó a 5 versos.")

    return out, warnings


# -------------------------
# OLLAMA: extracción robusta JSON
# -------------------------
def extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Extrae el primer objeto JSON { ... } aunque el modelo meta texto alrededor.
    """
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    i = text.find("{")
    j = text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError("No se encontró objeto JSON en respuesta.")
    candidate = text[i:j+1]
    return json.loads(candidate)

def ollama_rewrite(model: str, tipo: str, keyword: str, original_text: str, retries: int = 2) -> str:
    regla = REGLAS.get(tipo, REGLA_GENERIC)

    sys_msg = (
        "Eres un poeta experto en lengua española y formas poéticas.\n"
        "Tu salida DEBE ser JSON estricto (sin markdown, sin texto extra).\n"
        "Solo devuelve: {\"texto\": \"...\"}\n"
        f"Tipo objetivo: {tipo}\n"
        f"Regla técnica obligatoria: {regla}\n"
        f"Condición: Mantén el tema o palabra clave: {keyword!r}\n"
        f"El campo texto puede incluir saltos \\n para separar versos si aplica.\n"
        f"No cambies el tipo, solo reescribe el texto.\n"
    )

    user_msg = (
        "Reescribe este poema SOLO lo necesario para cumplir la forma indicada:\n\n"
        f"{original_text}\n"
    )

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                options={
                    "temperature": 0.35,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_ctx": 4096,
                    "num_predict": 900,
                }
            )
            content = (resp.get("message", {}) or {}).get("content", "") or ""
            data = extract_first_json_object(content)
            texto = str(data.get("texto", "") or "")
            texto = norm_spaces(fix_mojibake(texto))
            if not texto.strip():
                raise ValueError("JSON válido pero texto vacío.")
            return texto
        except Exception as e:
            last_err = e
            time.sleep(0.2 * (attempt + 1))

    raise RuntimeError(f"Ollama falló tras reintentos: {str(last_err)[:200]}")


# -------------------------
# Determinar tipo objetivo
# -------------------------
def infer_tipo(path: Path, schema: Dict[str, Any]) -> str:
    # 1) desde poema.tipo
    t = normalize_tipo(schema.get("poema", {}).get("tipo", ""))
    if t:
        if FORCE_ON_CONCRETE and t == "poema_concreto":
            return CONCRETE_FALLBACK_TIPO
        return t

    # 2) desde subcategoria
    t = normalize_tipo(schema.get("subcategoria", ""))
    if t:
        if FORCE_ON_CONCRETE and t == "poema_concreto":
            return CONCRETE_FALLBACK_TIPO
        return t

    # 3) desde nombre de archivo
    name = strip_accents(fix_mojibake(path.stem).lower())
    name = re.sub(r"[_\-]+", " ", name)
    for key in CANON:
        if key.endswith("_failed"):
            continue
        key_sp = key.replace("_", " ")
        if key in name or key_sp in name:
            if FORCE_ON_CONCRETE and key == "poema_concreto":
                return CONCRETE_FALLBACK_TIPO
            return key

    # 4) archivos guía
    if "guia" in name or name.startswith("00 "):
        return ""

    return ""


# -------------------------
# Guardar/mover
# -------------------------
def ensure_unique_path(dst_path: Path) -> Path:
    if not dst_path.exists():
        return dst_path
    base = dst_path.stem
    ext = dst_path.suffix
    parent = dst_path.parent
    for k in range(1, 10000):
        cand = parent / f"{base}__dup{k}{ext}"
        if not cand.exists():
            return cand
    raise RuntimeError("No se pudo encontrar nombre único para destino.")


# -------------------------
# MAIN: procesado por archivo
# -------------------------
def process_file(
    path: Path,
    dst_root: Path,
    model: str,
    dry_run: bool = False,
    report_include_text: bool = False,
) -> Dict[str, Any]:

    item: Dict[str, Any] = {
        "src": str(path),
        "status": "",
        "metodo": "",
        "tipo_objetivo": "",
        "dst": "",
        "warnings": [],
        "errores": [],
        "validacion_antes": {},
        "validacion_despues": {},
    }

    raw = safe_load_json(path)
    if raw is None or not isinstance(raw, dict):
        item["status"] = "failed"
        item["metodo"] = "parse_error"
        item["errores"].append("No se pudo leer/parsear JSON.")
        return item

    before = clean_to_schema(raw)

    # fix mojibake temprano
    before["subcategoria"] = fix_mojibake(before["subcategoria"])
    before["poema"]["texto"] = fix_mojibake(before["poema"]["texto"])
    before["poema"]["tipo"] = fix_mojibake(before["poema"]["tipo"])
    before["poema"]["palabra_clave_ingresada"] = fix_mojibake(before["poema"]["palabra_clave_ingresada"])

    # guardar schema en reporte (ligero por defecto)
    if report_include_text:
        item["antes_schema"] = before
    else:
        item["antes_schema"] = {
            "subcategoria": before.get("subcategoria", ""),
            "poema": {
                "tipo": before.get("poema", {}).get("tipo", ""),
                "palabra_clave_ingresada": before.get("poema", {}).get("palabra_clave_ingresada", ""),
                "texto_len": len(before.get("poema", {}).get("texto", "") or ""),
            }
        }

    # tipo base por metadata/nombre
    tipo = infer_tipo(path, before)
    if not tipo:
        item["status"] = "skipped"
        item["metodo"] = "no_tipo"
        item["errores"].append("No se pudo determinar tipo objetivo.")
        return item

    # NUEVO: si el texto parece concreto/visual -> forzar tipo
    if FORCE_ON_CONCRETE:
        txt0 = before.get("poema", {}).get("texto", "") or ""
        if detect_poema_concreto_visual(txt0):
            if tipo != CONCRETE_FALLBACK_TIPO:
                item["warnings"].append(
                    f"Detectó poema_concreto/visual por heurística -> forzó tipo a {CONCRETE_FALLBACK_TIPO}."
                )
            tipo = CONCRETE_FALLBACK_TIPO

    item["tipo_objetivo"] = tipo

    # valida antes
    v_before = validate(tipo, before["poema"]["texto"])
    item["validacion_antes"] = v_before

    # 1) conservador
    after, warns = conservative_repair(tipo, before)
    item["warnings"].extend(warns)

    v_after = validate(tipo, after["poema"]["texto"])

    # 2) si no pasa, Ollama
    if not v_after.get("ok", False):
        item["metodo"] = "ollama"
        try:
            keyword = after["poema"].get("palabra_clave_ingresada", "") or after.get("subcategoria", "") or ""
            new_text = ollama_rewrite(model=model, tipo=tipo, keyword=keyword, original_text=after["poema"]["texto"])
            after["poema"]["texto"] = new_text
            v_after = validate(tipo, after["poema"]["texto"])
            if not v_after.get("ok", False):
                item["errores"].append(f"No pasó validación tras Ollama: {v_after.get('errores', [])}")
        except Exception as e:
            item["errores"].append(f"Error Ollama: {str(e)[:200]}")
    else:
        item["metodo"] = "conservador"

    item["validacion_despues"] = v_after

    # schema final en reporte
    if report_include_text:
        item["despues_schema"] = after
    else:
        item["despues_schema"] = {
            "subcategoria": after.get("subcategoria", ""),
            "poema": {
                "tipo": after.get("poema", {}).get("tipo", ""),
                "palabra_clave_ingresada": after.get("poema", {}).get("palabra_clave_ingresada", ""),
                "texto_len": len(after.get("poema", {}).get("texto", "") or ""),
            }
        }

    # si aún falla -> failed
    if not v_after.get("ok", False):
        item["status"] = "failed"
        failed_dir = dst_root / CANON["complementaria_failed"]
        dst_path = ensure_unique_path(failed_dir / path.name)
        item["dst"] = str(dst_path)

        if not dry_run:
            failed_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dst_path))
        return item

    # ok -> escribir schema corregido y borrar/mover original
    item["status"] = "ok"
    dst_dir = dst_root / CANON.get(tipo, tipo)
    dst_path = ensure_unique_path(dst_dir / path.name)
    item["dst"] = str(dst_path)

    if not dry_run:
        atomic_write_json(dst_path, after)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            try:
                backup = path.with_suffix(path.suffix + ".moved")
                shutil.move(str(path), str(backup))
                item["warnings"].append("No se pudo borrar original; se renombró a .moved")
            except Exception:
                item["warnings"].append("No se pudo borrar ni mover el original.")

    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3", help="Modelo de Ollama (ej: gemma3)")
    ap.add_argument("--src", default=str(SRC_DIR_DEFAULT), help="Carpeta origen (complementaria)")
    ap.add_argument("--dst", default=str(DST_DIR_DEFAULT), help="Carpeta destino (dataset_final_validado)")
    ap.add_argument("--report", default=str(ROOT / "reporte_reclasificacion.json"), help="Ruta del reporte JSON")
    ap.add_argument("--limit", type=int, default=0, help="Limitar número de archivos (0 = todos)")
    ap.add_argument("--dry-run", action="store_true", help="No mueve ni escribe, solo simula y genera reporte")
    ap.add_argument("--report-include-text", action="store_true", help="Incluye texto completo antes/después en reporte (pesado).")
    args = ap.parse_args()

    model = args.model
    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    report_path = Path(args.report).resolve()
    dry_run = args.dry_run
    report_include_text = args.report_include_text

    if not src_root.exists():
        print(f"ERROR: Origen no existe: {src_root}")
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / CANON["complementaria_failed"]).mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src_root.glob("*.json") if p.is_file()])
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    print(f"Modelo: {model}")
    print(f"Origen: {src_root}")
    print(f"Destino: {dst_root}")
    print(f"Archivos a procesar: {len(files)}")
    print(f"Reporte: {report_path}")
    if dry_run:
        print("MODO: DRY-RUN (no se escribe nada)")
    if report_include_text:
        print("Reporte: incluye texto completo (PESADO)")

    ok = failed = skipped = 0
    items: List[Dict[str, Any]] = []

    for i, path in enumerate(files, start=1):
        try:
            item = process_file(
                path,
                dst_root,
                model=model,
                dry_run=dry_run,
                report_include_text=report_include_text,
            )
            items.append(item)

            if item["status"] == "ok":
                ok += 1
            elif item["status"] == "failed":
                failed += 1
            else:
                skipped += 1

            if i % 10 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] ok={ok} failed={failed} skipped={skipped}")

        except Exception as e:
            failed += 1
            items.append({
                "src": str(path),
                "status": "failed",
                "metodo": "exception",
                "tipo_objetivo": "",
                "dst": "",
                "warnings": [],
                "errores": [f"Excepción: {str(e)[:220]}"],
                "validacion_antes": {},
                "validacion_despues": {},
            })

    report = {
        "model": model,
        "src": str(src_root),
        "dst": str(dst_root),
        "total": len(files),
        "ok": ok,
        "failed": failed,
        "skipped": skipped,
        "items": items
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- FIN ---")
    print(f"OK: {ok}")
    print(f"FAILED: {failed} (movidos a {dst_root / CANON['complementaria_failed']})")
    print(f"SKIPPED: {skipped}")
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()
