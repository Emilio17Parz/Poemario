import json
import zipfile
import io

# Lista de datos para los 10 Haikus
datos_haikus = [
    {"kw": "Luna", "sub": "Naturaleza", "txt": "Luz en la noche,\nla luna nos observa,\nduerme el jardín."},
    {"kw": "Río", "sub": "Paisaje", "txt": "Agua que corre,\nsusurros de cristal,\nvida que pasa."},
    {"kw": "Otoño", "sub": "Estaciones", "txt": "Hojas doradas,\nbailan con el viento frío,\nel suelo cruje."},
    {"kw": "Flor", "sub": "Primavera", "txt": "Brota la vida,\npétalos de colores,\nsol en la rama."},
    {"kw": "Nieve", "sub": "Invierno", "txt": "Manto de blanco,\nsilencio en la montaña,\nfrío que abraza."},
    {"kw": "Mar", "sub": "Océano", "txt": "Olas gigantes,\nrompen en la orilla gris,\nsal en el aire."},
    {"kw": "Gato", "sub": "Animales", "txt": "Pasos callados,\nojos como luceros,\nrey de la casa."},
    {"kw": "Café", "sub": "Cotidiano", "txt": "Aroma oscuro,\ndespierta la mañana,\ncalor en manos."},
    {"kw": "Lluvia", "sub": "Clima", "txt": "Golpean gotas,\nmúsica en el tejado,\ngris horizonte."},
    {"kw": "Cerezo", "sub": "Botánica", "txt": "Rosa y blanco,\nflor efímera y bella,\ncae la tarde."}
]

nombre_zip = "coleccion_haikus.zip"

print(f"Generando {nombre_zip}...")

with zipfile.ZipFile(nombre_zip, 'w') as zf:
    for i, data in enumerate(datos_haikus, 1):
        # Estructura solicitada por el usuario
        contenido_json = {
            "subcategoria": data["sub"],
            "poema": {
                "texto": data["txt"],
                "tipo": "Haiku",
                "palabra_clave_ingresada": data["kw"]
            }
        }
        
        # Nombre del archivo individual (ej: haiku_1_Luna.json)
        nombre_archivo = f"haiku_{i}_{data['kw']}.json"
        
        # Convertir a string JSON con formato bonito (indent=4) y caracteres especiales (ensure_ascii=False)
        json_str = json.dumps(contenido_json, indent=4, ensure_ascii=False)
        
        # Escribir en el zip
        zf.writestr(nombre_archivo, json_str)

print("¡Listo! Se ha creado el archivo 'coleccion_haikus.zip' con 10 JSONs.")