import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from collections import Counter

# --- CONFIGURACIÓN ---
BASE_DIR = r"C:\Users\jecal\Poemario"
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

# URL base para completar links relativos
BASE_URL = "https://www.poemas-del-alma.com"

# LISTA DE AUTORES A MINAR (Puedes agregar más URLs aquí)
URLS_AUTORES = [
    "https://www.poemas-del-alma.com/pablo-neruda.htm",
    "https://www.poemas-del-alma.com/ruben-dario.htm",
    "https://www.poemas-del-alma.com/mario-benedetti.htm",
    "https://www.poemas-del-alma.com/federico-garcia-lorca.htm",
    "https://www.poemas-del-alma.com/gabriela-mistral.htm",
    "https://www.poemas-del-alma.com/antonio-machado.htm",
    "https://www.poemas-del-alma.com/miguel-de-unamuno.htm"
]

# --- (AQUÍ REUTILIZAMOS TU LÓGICA DE CLASIFICACIÓN PREVIA) ---
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
            diph = len(re.findall(r'[aeoáéó][iuü]|[iuü][aeoáéó]|[iuü][iuü]', word))
            n = max(1, n - diph)
            starts_vowel = word[0] in (self.strong + self.weak)
            if last_ends_vowel and starts_vowel: sinalefas += 1
            last_ends_vowel = word[-1] in (self.strong + self.weak)
            raw += n
        last_word = words[-1]
        adj = 0
        if last_word[-1] in "ns" or last_word[-1] in (self.strong + self.weak):
            if re.search(r'[áéíóú].*[aeiou][ns]?', last_word): adj = -1
        else: adj = 1
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

    def analyze_structure(self, lines):
        count = len(lines)
        if count == 0: return None
        syllables = [self.analyzer.count_syllables(l) for l in lines]
        avg_syl = sum(syllables) / count
        endings = [self.analyzer.get_rhyme_ending(l) for l in lines]
        is_arte_mayor = 9.5 <= avg_syl <= 12.5
        is_arte_menor = 6.5 <= avg_syl <= 9.0
        
        if count == 3 and 4<=syllables[0]<=6: return "haiku", "Métrica 5-7-5"
        if count == 14 and is_arte_mayor: return "soneto", "14 versos endecasílabos"
        if count == 10 and is_arte_menor: return "decima", "10 versos octosílabos"
        if count == 4:
            r_abba = endings[0] == endings[3] and endings[1] == endings[2]
            r_abab = endings[0] == endings[2] and endings[1] == endings[3]
            if is_arte_mayor and r_abba: return "cuarteto", "Cuarteto ABBA"
            if is_arte_mayor and r_abab: return "serventesio", "Serventesio ABAB"
            if is_arte_menor and r_abba: return "redondilla", "Redondilla abba"
            if is_arte_menor and r_abab: return "cuarteta", "Cuarteta abab"
        if count > 8 and is_arte_menor:
            assonants = [self.analyzer.get_assonant_ending(l) for l in lines]
            evens = assonants[1::2]
            if evens:
                most = Counter(evens).most_common(1)
                if most and most[0][1] > len(evens)*0.5: return "romance", "Romance asonante"
        return None

    def analyze_semantic(self, full_text):
        text_lower = full_text.lower()
        scores = {k: 0 for k in KEYWORDS.keys()}
        found = {}
        for cat, kws in KEYWORDS.items():
            for kw in kws:
                if f" {kw} " in text_lower:
                    scores[cat] += 1
                    found[cat] = kw
        best = max(scores, key=scores.get)
        if scores[best] > 0: return best, found[best]
        if len(full_text.split()) > 400: return "narrativo", "Extensión larga"
        lines = full_text.split('\n')
        avg = sum(len(l) for l in lines)/len(lines) if lines else 0
        if avg > 75: return "prosa", "Prosa poética"
        return "verso_libre", "Sin estructura fija"

# --- CLASE DEL MINERO WEB ---

class WebMiner:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.classifier = PoemClassifier()
        self.extracted_count = 0

    def get_soup(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                # Importante: decodificar correctamente acentos
                response.encoding = response.apparent_encoding
                return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"❌ Error conectando a {url}: {e}")
        return None

    def clean_text(self, soup_element):
        """
        Limpia el HTML del poema según tus screenshots:
        1. Elimina scripts y estilos.
        2. Elimina div.likebox (publicidad).
        3. Convierte <br> en saltos de línea \n.
        """
        if not soup_element: return ""
        
        # Eliminar basura conocida
        for tag in soup_element(['script', 'style', 'iframe']):
            tag.decompose()
            
        # Eliminar específicamente el likebox (screenshot 2)
        likebox = soup_element.find('div', class_='likebox')
        if likebox:
            likebox.decompose()
            
        # Reemplazar <br> con \n para que .get_text() no pegue las palabras
        for br in soup_element.find_all("br"):
            br.replace_with("\n")
            
        return soup_element.get_text().strip()

    def process_poem_page(self, poem_url, author_name):
        soup = self.get_soup(poem_url)
        if not soup: return

        # Selector basado en screenshot 3: div.poem-entry#contentfont > p
        content_div = soup.find("div", {"class": "poem-entry", "id": "contentfont"})
        
        if not content_div:
            # Fallback: a veces solo es .poem-entry sin id
            content_div = soup.find("div", class_="poem-entry")
        
        if not content_div:
            print(f"⚠️  No se encontró contenido en {poem_url}")
            return

        # Buscar el párrafo <p> que contiene el poema
        p_tag = content_div.find('p')
        if not p_tag:
            # A veces el texto está directo en el div
            p_tag = content_div

        raw_text = self.clean_text(p_tag)
        
        # Filtrar si está vacío o muy corto
        if len(raw_text) < 20: return

        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        full_text = "\n".join(lines)

        # --- CLASIFICACIÓN ---
        result = self.classifier.analyze_structure(lines)
        if not result:
            result = self.classifier.analyze_semantic(full_text)
        
        cat_code, reason = result
        subcat_folder = FOLDERS.get(cat_code, "Verso libre")

        # --- GUARDAR JSON ---
        self.save_json(full_text, cat_code, subcat_folder, reason, poem_url, author_name)
        print(f"✅ Guardado: {subcat_folder} | {reason}")

    def save_json(self, text, type_code, subcategory, keyword, url, author):
        target_dir = os.path.join(DATASET_DIR, subcategory)
        os.makedirs(target_dir, exist_ok=True)
        
        # Crear nombre de archivo único basado en la URL
        slug = url.split('/')[-1].replace('.htm', '')
        filename = f"web_{slug}.json"
        
        data = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "subcategoria": subcategory,
            "poema": {
                "texto": text,
                "tipo": type_code,
                "palabra_clave_ingresada": keyword,
                "autor": author, # Extra útil para web
                "origen": url
            }
        }
        
        with open(os.path.join(target_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.extracted_count += 1

    def run_author(self, author_url):
        print(f"\n🌍 Procesando autor: {author_url}")
        soup = self.get_soup(author_url)
        if not soup: return
        
        # Extraer nombre autor del título o URL
        author_name = author_url.split('/')[-1].replace('.htm', '').replace('-', ' ').title()

        # Selector basado en screenshot 1: ul#ordenable li a
        poem_list = soup.find("ul", id="ordenable")
        if not poem_list:
            print("No se encontró lista de poemas.")
            return

        links = poem_list.find_all("a")
        print(f"📚 Encontrados {len(links)} poemas. Iniciando descarga...")

        for link in links:
            href = link.get('href')
            if not href: continue
            
            # Construir URL absoluta
            if href.startswith('http'):
                full_url = href
            else:
                full_url = f"{BASE_URL}/{href}"
            
            # Pausa de cortesía (importante para no ser bloqueado)
            time.sleep(1) 
            self.process_poem_page(full_url, author_name)

    def start(self):
        print("🚀 INICIANDO MINERO WEB")
        for url in URLS_AUTORES:
            self.run_author(url)
        print(f"\n🏁 Proceso finalizado. Total poemas web extraídos: {self.extracted_count}")

if __name__ == "__main__":
    miner = WebMiner()
    miner.start()