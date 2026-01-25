#  Poemario IA

Repositorio colaborativo para la construcción del mayor dataset estructurado de poesía generada con inteligencia artificial en español.

##  Objetivo

Crear un repositorio validado, sin duplicados, con poemas clasificados por forma poética, subcategoría y palabra detonante, para entrenar modelos generativos capaces de producir poemas coherentes, métricos e ilustrados.

##  Estructura del proyecto

datasets/<tipo>/<subcategoria>.json
schema/poema.schema.json
scripts/validator.py
pipeline/github-actions.yml

##  Cómo agregar nuevos poemas al Poemario IA

###  Instalar Git (Windows Automático)

Pega esto en PowerShell y presiona ENTER:

Invoke-WebRequest -Uri https://github.com/git-for-windows/git/releases/download/v2.45.0.windows.1/Git-2.45.0-64-bit.exe -OutFile git.exe
.\git.exe /VERYSILENT /NORESTART
setx PATH "$env:PATH;C:\Program Files\Git\cmd;C:\Program Files\Git\bin"
git --version
git clone https://github.com/Emilio17Parz/Poemario.git


## 2 Crear una rama nueva
bash
Copy code
git checkout -b agregar-poema


## 3 Elegir tipo de poema
Los poemas se guardan en la carpeta correspondiente dentro de datasets/:


datasets/Soneto/
datasets/Haiku/
datasets/Poema satirico/
...

cd "nombre de la carpeta"
## 4 Crear el JSON del poema
Debe seguir este formato:


## ejemplo
{
  "subcategoria": "amor prohibido",
  "poema": {
    "titulo": "El fuego y la sombra",
    "texto": "Arde tu luz en néctar encendido...",
    "tipo": "Soneto",
    "palabra_clave_ingresada": "fuego"
  }
}

## 5 Validar (opcional)


python scripts/validator.py

## 6️ Enviar cambios


git add .
git commit -m "Agregar nuevo poema"
git push origin agregar-poema


## 7 Crear Pull Request en GitHub

## 3. SCRIPT PARA CREAR JSON VÁLIDOS AUTOMÁTICAMENTE**

Guarda en:

scripts/create_poema.py

python
Copy code

```python
import json
import sys
import os

TIPO = sys.argv[1]
SUB = sys.argv[2]
TIT = sys.argv[3]
PAL = sys.argv[4]

texto = input("Pega el poema completo (con saltos de línea):\n")

data = {
    "subcategoria": SUB,
    "poema": {
        "titulo": TIT,
        "texto": texto.replace("\n", "\\n"),
        "tipo": TIPO,
        "palabra_clave_ingresada": PAL
    }
}

os.makedirs(f"datasets/{TIPO}", exist_ok=True)

filename = f"datasets/{TIPO}/{SUB.replace(' ','_')}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Archivo creado: {filename}")
Uso:

bash
Copy code
python scripts/create_poema.py Soneto "amor prohibido" "El fuego y la sombra" fuego

## 4. COMMIT LISTO PARA PEGAR
En tu repo local:

bash
Copy code
git add schema/poema.schema.json scripts/validator.py README.md scripts/create_poema.py
git commit -m "Refactor JSON structure, remove metadata, add poem generator, update docs"
git push origin dev


