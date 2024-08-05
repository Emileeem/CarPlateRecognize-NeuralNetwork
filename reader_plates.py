import os
import json
import easyocr
import cv2
import numpy as np
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

folder_path = "./placas"
json_file = 'indicePlacas.json'

reader = easyocr.Reader(['pt'])

# Mapeamento de substituições comuns que a IA costuma confundir
substitution_map = {
    'A': '4',
    '4': 'A',
    'O': '0',
    '0': 'O',
    'D': '0',
    '0': 'D',
    'W': 'M',
    'M': 'W',
    '1': 'I',
    'I': '1',
    "'": ""
}

# Aplicação das substituições no texto
def apply_substitutions(text):
    for key, value in substitution_map.items():
        text = text.replace(key, value)
    return text

# Carrega os dados do arquivo JSON de placas
def load_plates_from_json(json_file):
    with open(json_file, 'r') as f:
        plates_data = json.load(f)
    return {plate['indice']: plate['placa'] for plate in plates_data}

plates_data = load_plates_from_json(json_file)

image_files = os.listdir(folder_path)

for filename in image_files:
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)

        result = reader.readtext(image_path)
        detected_text = result[-1][1] if result else ""  # Obtém o último texto detectado

        print(f"Texto Lido na Imagem {filename}: {detected_text}")  # Exibe o texto lido pelo OCR

        index_str = filename.split('_')[1].split('.')[0]

        try:
            index = int(index_str)
        except ValueError:
            print(f"Erro: Não foi possível extrair o índice do arquivo {filename}")
            continue

        real_text = plates_data.get(index)
        if real_text is None:
            print(f"Placa para o arquivo {filename} não encontrada no JSON.")
            continue

        print(f"Texto Real da Placa: {real_text}")

        # Aplica as substituições comuns nos textos detectado e real
        detected_text_normalized = apply_substitutions(detected_text.upper())
        real_text_normalized = apply_substitutions(real_text.upper())

        # Verifica se o texto detectado corresponde ao texto real da placa
        if detected_text_normalized == real_text_normalized:
            print("Resultado CORRETO!")
        else:
            print("Resultado INCORRETO!")
            print("Verifique a placa manualmente.")


cv2.destroyAllWindows()
