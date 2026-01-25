import google.generativeai as genai
from google.api_core import exceptions

# ================= PEGA TUS API KEYS AQUÍ =================
API_KEYS = [
    "AIzaSyA-eykTdtxObSFheIIRs5Y5T2lNnTLomns",
    "AIzaSyCo_sTXqEfHIgL0RUqr1qlao6nDuyYeank",
    "AIzaSyDQPZPqg2vImRkP14eNuf8HGiVCsS2HtJE",
    "AIzaSyDCvjfWx2A8eNYYJWugZ6sNRNHrsnxfLvU",
    "AIzaSyBt0JH35K-eOFur__gYzh3LPX3PtF5J39U",
    "AIzaSyB9HTBA941Ox4S-GbvZaXTH8IYAK4C-hpE",
    "AIzaSyDlU4bmcj7ABgzeRlcc_ZcNBupr94TSx7c"
]
# =========================================================

def verificar_llaves():
    print("--- INICIANDO VERIFICACIÓN DE MODELOS DISPONIBLES ---\n")

    for index, key in enumerate(API_KEYS):
        print(f"🔹 Probando API KEY #{index + 1} (Termina en ...{key[-4:]})")
        
        try:
            genai.configure(api_key=key)
            
            # Listar modelos disponibles
            modelos = list(genai.list_models())
            
            # Filtrar solo los que sirven para generar texto (generateContent)
            modelos_generativos = [
                m for m in modelos 
                if 'generateContent' in m.supported_generation_methods
            ]

            if not modelos_generativos:
                print("   ⚠️  La llave es válida, pero no tiene modelos con 'generateContent'.")
            else:
                print("   ✅ Modelos disponibles para generar texto:")
                for m in modelos_generativos:
                    print(f"      - {m.name}")
        
        except exceptions.PermissionDenied:
            print("   ❌ Error: Permiso denegado (API Key inválida o expirada).")
        except exceptions.ResourceExhausted:
            print("   ❌ Error: Cuota excedida (Quota limit reached).")
        except Exception as e:
            print(f"   ❌ Error desconocido: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    verificar_llaves()