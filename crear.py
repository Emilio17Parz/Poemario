#!/usr/bin/env python
import os

# Lista de carpetas normalizadas
carpetas = ['Soneto', 'Haiku', 'Tanka', 'Limerick', 'Oda', 'Elegia', 'Egloga', 'Epigrama', 'Romance', 'Decima_espinela', 'Redondilla', 'Cuarteta', 'Cuarteto', 'Serventesio', 'Terceto', 'Terceto_encadenado', 'Pareado', 'Silva', 'Copla', 'Seguidilla', 'Estrofa_safica', 'Estrofa_alcaica', 'Estancia', 'Balada', 'Villanelle', 'Sestina', 'Pantoum', 'Rondo', 'Rondeau', 'Triolet', 'Madrigal', 'Zejel', 'Moaxaja', 'Gacela_ghazal', 'Cancion_petrarquista', 'Himno', 'Poema_en_prosa', 'Verso_libre', 'Versiculo', 'Acrostico', 'Palindromo_poetico', 'Poema_concreto', 'Poema_narrativo', 'Poema_dramatico', 'Poema_lirico', 'Poema_elegiaco', 'Poema_epico', 'Poema_satirico', 'Poema_didactico']

for carpeta in carpetas:
    ruta = os.path.join('datasets', carpeta)
    os.makedirs(ruta, exist_ok=True)
    gitkeep = os.path.join(ruta, '.gitkeep')
    if not os.path.exists(gitkeep):
        with open(gitkeep, 'w', encoding='utf-8') as f:
            f.write('')
print('✅ Se han creado todas las carpetas poéticas con .gitkeep')