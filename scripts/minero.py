import os
import re
import json
from collections import Counter

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = r"C:\Users\jecal\Poemario"
RAW_DIR = os.path.join(BASE_DIR, "libros_raw")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

# --- MAPEO DE CLASES (Identificador -> Carpeta) ---
FOLDERS = {
    # Estructuras Fijas
    "soneto": "soneto",
    "haiku": "Haiku",
    "tanka": "Tanka",
    "decima": "Decima espinela",
    "lira": "Lira",
    "romance": "Romance",
    "cuarteto": "Cuarteto",
    "redondilla": "Redondilla",
    "serventesio": "Serventesio",
    "cuarteta": "Cuarteta",
    "copla": "Copla",
    "terceto": "Terceto",
    "terceto_encadenado": "Terceto encadenado",
    "pareado": "Pareado",
    "silva": "Silva",
    "estancia": "Estancia",
    "madrigal": "Madrigal",
    "limerick": "Limerick",
    "villanelle": "Villanelle",
    "sestina": "Sestina",
    "rondeau": "Rondeau",
    "rondo": "Rondo",
    "triolet": "Triolet",
    "pantoum": "Pantoum",
    "zejel": "Zejel",
    "moaxaja": "Moaxaja",
    "gacela": "Gacela",
    "ghazal": "ghazal",
    "seguidilla": "Seguidilla",
    "estrofa_safica": "Estrofa safica",
    "estrofa_alcaica": "Estrofa alcaica",
    
    # Géneros
    "elegia": "Elegia",
    "oda": "Oda",
    "himno": "Himno",
    "egloga": "Egloga",
    "epigrama": "Epigrama",
    "satira": "Poema satirico",
    "epico": "Poema epico", 
    "narrativo": "Poema narrativo",
    "dramatico": "Poema dramatico",
    "didactico": "Poema didactico",
    "lirico": "Poema lirico",
    "prosa": "Poema en prosa",
    "versiculo": "Versiculo",
    
    # Temáticas
    "amor_triste": "Desamor_tristeza_perdida",
    "religion": "Religion_espiritualidad",
    "social": "Sociedad_critica_social",
    "vida": "Vida_y_existencia",
    "yo": "Yo_interior_introspeccion",
    "aventura": "Aventura_epica_heroismo",
    
    # Visuales / Experimentales
    "acrostico": "Acrostico",
    "caligrama": "caligrama",
    "concreto": "Poema concreto",
    "palindromo": "Palindromo poetico",
    
    # Default
    "verso_libre": "Verso libre"
}

KEYWORDS = {
    "religion": ["dios", "cielo", "alma", "rezo", "santo", "cristo", "fe", "plegaria", "espíritu", "iglesia", "pecado"],
    "amor_triste": ["llanto", "tristeza", "dolor", "adiós", "muerte", "tumba", "lágrima", "soledad", "perdido", "ausencia", "pena"],
    "social": ["pueblo", "patria", "libertad", "tirano", "guerra", "injusticia", "pobre", "obrero", "nación", "bandera"],
    "vida": ["tiempo", "vida", "muerte", "vejez", "destino", "camino", "existencia", "mundo", "fugaz", "reloj"],
    "yo": ["sueño", "mente", "pensamiento", "yo", "mismo", "espejo", "interior", "conciencia"],
    "aventura": ["espada", "batalla", "gloria", "héroe", "caballo", "honor", "victoria", "mar", "viaje"],
    "egloga": ["pastor", "oveja", "prado", "campo", "hierba", "ganado", "flauta"],
    "oda": ["oda", "loor", "alabanza", "canto a"],
    "himno": ["himno", "gloria", "salve"],
    "epigrama": ["epigrama"]
}

class MetricAnalyzer:
    def __init__(self):
        self.strong = "aeoáéó"
        self.weak = "iuüíú"
    
    def count_syllables(self, verse):
        verse = verse.lower()
        words = re.findall(r'\b\w+\b', verse)
        if not words: return 0
        raw, sinalefas = 0, 0
        last_ends_vowel = False
        
        for word in words:
            vowels = [c for c in word if c in self.strong + self.weak]
            n = len(vowels)
            # Diptongos aprox
            diph = len(re.findall(r'[aeoáéó][iuü]|[iuü][aeoáéó]|[iuü][iuü]', word))
            n = max(1, n - diph)
            
            starts_vowel = word[0] in (self.strong + self.weak)
            if last_ends_vowel and starts_vowel: sinalefas += 1
            last_ends_vowel = word[-1] in (self.strong + self.weak)
            raw += n
            
        # Acento final
        last_word = words[-1]
        adj = 0
        if last_word[-1] in "ns" or last_word[-1] in (self.strong + self.weak):
            if re.search(r'[áéíóú].*[aeiou][ns]?', last_word): adj = -1
        else:
            adj = 1
        return max(1, raw - sinalefas + adj)

    def get_rhyme_ending(self, verse):
        v = verse.lower().strip()
        return v[-3:] if len(v) >= 3 else v

    def get_assonant_ending(self, verse):
        v = verse.lower()
        vowels = [c for c in v if c in "aeiouáéíóú"]
        return "".join(vowels[-2:]) if len(vowels) >= 2 else "".join(vowels)

class PoemClassifier:
    def __init__(self):
        self.analyzer = MetricAnalyzer()
        self.extracted = 0

    def analyze_structure(self, lines):
        """Retorna (categoria, razon_detectada) o None."""
        count = len(lines)
        if count == 0: return None
        
        syllables = [self.analyzer.count_syllables(l) for l in lines]
        avg_syl = sum(syllables) / count
        endings = [self.analyzer.get_rhyme_ending(l) for l in lines]
        
        is_arte_mayor = 9.5 <= avg_syl <= 12.5
        is_arte_menor = 6.5 <= avg_syl <= 9.0
        
        # HAIKU / TANKA
        if count == 3 and 4<=syllables[0]<=6 and 6<=syllables[1]<=8 and 4<=syllables[2]<=6:
            return "haiku", "Estructura métrica 5-7-5"
        if count == 5 and syllables[:3] == [5,7,5]:
            return "tanka", "Estructura métrica 5-7-5-7-7"

        # SONETO / DECIMA
        if count == 14 and is_arte_mayor:
            return "soneto", "14 versos endecasílabos"
        if count == 10 and is_arte_menor:
            return "decima", "10 versos octosílabos"

        # FORMAS COMPLEJAS
        if count == 19: return "villanelle", "Estructura 19 versos"
        if count == 39: return "sestina", "Estructura 39 versos"
        if count == 15: return "rondeau", "Estructura 15 versos"
        
        # TRIOLET / LIMERICK
        if count == 8 and endings[0] == endings[3]:
            return "triolet", "8 versos con repetición"
        if count == 5 and endings[0]==endings[1]==endings[4] and endings[2]==endings[3]:
            if syllables[2] < syllables[0]:
                return "limerick", "Rima AABBA rítmica"

        # ESTROFAS CORTAS
        if count == 2:
            return "pareado", "2 versos (Pareado)"
        if count == 3:
            return "terceto", "3 versos (Terceto)"
        
        if count == 4:
            r_abba = endings[0] == endings[3] and endings[1] == endings[2]
            r_abab = endings[0] == endings[2] and endings[1] == endings[3]
            
            if is_arte_mayor:
                if r_abba: return "cuarteto", "4 versos rima ABBA"
                if r_abab: return "serventesio", "4 versos rima ABAB"
                return "cuarteto", "4 versos arte mayor"
            elif is_arte_menor:
                if r_abba: return "redondilla", "4 versos rima abba"
                if r_abab: return "cuarteta", "4 versos rima abab"
                return "redondilla", "4 versos arte menor"

        # ROMANCE
        if count > 8 and is_arte_menor:
            assonants = [self.analyzer.get_assonant_ending(l) for l in lines]
            evens = assonants[1::2]
            if evens:
                most_common = Counter(evens).most_common(1)
                if most_common and most_common[0][1] > len(evens) * 0.5:
                    return "romance", f"Romance (Asonancia '{most_common[0][0]}')"
        
        # SILVA
        has_7 = any(6 <= s <= 8 for s in syllables)
        has_11 = any(10 <= s <= 12 for s in syllables)
        if count > 6 and has_7 and has_11:
            return "silva", "Combinación silva (7 y 11)"

        return None

    def analyze_semantic(self, full_text):
        """Retorna (categoria, palabra_clave_encontrada)."""
        text_lower = full_text.lower()
        
        if "acróstico" in text_lower: 
            return "acrostico", "Título/Texto contiene 'acróstico'"
            
        scores = {k: 0 for k in KEYWORDS.keys()}
        found_word = {}
        
        for cat, kws in KEYWORDS.items():
            for kw in kws:
                if f" {kw} " in text_lower or f" {kw}," in text_lower:
                    scores[cat] += 1
                    found_word[cat] = kw # Guardar cuál palabra detonó
        
        best_cat = max(scores, key=scores.get)
        if scores[best_cat] > 0:
            return best_cat, found_word[best_cat]
            
        # Clasificación por longitud si no hay keywords
        if len(full_text.split()) > 400:
            return "narrativo", "Extensión larga (>400 palabras)"
        
        lines = full_text.split('\n')
        avg_len_chars = sum(len(l) for l in lines) / len(lines) if lines else 0
        if avg_len_chars > 75:
            return "prosa", "Líneas muy largas (Prosa poética)"

        return "verso_libre", "Sin estructura fija detectada"

    def process_file(self, filepath):
        print(f"Minando: {os.path.basename(filepath)}...")
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except: return

        blocks = re.split(r'\n\s*\n', content)
        
        for i, block in enumerate(blocks):
            lines = [l.strip() for l in block.split('\n') if l.strip()]
            if len(lines) < 2: continue
            if "Gutenberg" in block: continue
            
            full_text = "\n".join(lines)
            
            # 1. Intentar Estructura
            result = self.analyze_structure(lines)
            
            # 2. Si falla, Semántica
            if not result:
                result = self.analyze_semantic(full_text)
            
            category_code, reason = result
            
            # 3. Guardar JSON
            folder_name = FOLDERS.get(category_code, "Verso libre")
            self.save_json_poem(full_text, category_code, folder_name, reason, filepath, i)

    def save_json_poem(self, text, type_code, subcategory, keyword, source_path, index):
        target_dir = os.path.join(DATASET_DIR, subcategory)
        os.makedirs(target_dir, exist_ok=True)
        
        # Construir nombre archivo
        src_name = os.path.splitext(os.path.basename(source_path))[0]
        src_name = re.sub(r'[^\w\-]', '_', src_name)
        filename = f"{src_name}_{index}.json"
        
        # ESTRUCTURA JSON REQUERIDA
        data = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "subcategoria": subcategory,
            "poema": {
                "texto": text,
                "tipo": type_code,
                "palabra_clave_ingresada": keyword
            }
        }
        
        with open(os.path.join(target_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        self.extracted += 1

if __name__ == "__main__":
    classifier = PoemClassifier()
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.txt')]
    
    print(f"--- INICIANDO MINERÍA JSON ---")
    for f in files:
        classifier.process_file(os.path.join(RAW_DIR, f))
        
    print(f"\nProceso terminado. Total JSONs generados: {classifier.extracted}")