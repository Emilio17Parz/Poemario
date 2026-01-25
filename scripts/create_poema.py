import json
import os

def listar_tipos():
    rutas = os.listdir("datasets2")
    tipos = [d for d in rutas if os.path.isdir(os.path.join("datasets", d))]
    return sorted(tipos)

def main():
    print("📘 GENERADOR DE POEMAS JSON (SIN TÍTULO)")

    tipos = listar_tipos()
    print("\n📂 TIPOS DISPONIBLES:\n")
    for i, t in enumerate(tipos, 1):
        print(f"{i}. {t}")

    indice = int(input("\nSelecciona un número de tipo de poema: "))
    TIPO = tipos[indice - 1]

    SUB = input("Subcategoría (ej: amor prohibido): ")
    PAL = input("Palabra clave ingresada por el usuario: ")

    print("\nPega el poema verso por verso.")
    print("Escribe FIN para terminar.")
    print("----------------------------------------------")

    lineas = []
    while True:
        linea = input()
        if linea.strip() == "FIN":
            break
        lineas.append(linea)

    texto = "\\n".join(lineas)

    data = {
        "subcategoria": SUB,
        "poema": {
            "texto": texto,
            "tipo": TIPO,
            "palabra_clave_ingresada": PAL
        }
    }

    carpeta = f"datasets/{TIPO}"
    os.makedirs(carpeta, exist_ok=True)

    nombre = SUB.replace(" ", "_").lower()
    filename = f"{carpeta}/{nombre}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Poema creado con éxito:")
    print(f"📄 Archivo: {filename}")

if __name__ == "__main__":
    main()
