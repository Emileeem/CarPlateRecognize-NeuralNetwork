import os
import json
import easyocr
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

folder_path = "./placas"

reader = easyocr.Reader(['pt'])

# Mapeamento de substituições comuns que a ia costuma confundir
substitution_map = {
    'A': '4',
    '4': 'A',
    'O': '0',
    '0': 'O',
    'D': '0',
    '0': 'D',
    'O': 'D',
    'D': 'O',
    'W': 'M',
    'M': 'W',

}
def apply_substitutions(text):
    for key, value in substitution_map.items():
        text = text.replace(key, value)
    return text

# Carregar dados do arquivo JSON
def load_plates_from_json(json_file):
    with open(json_file, 'r') as f:
        plates_data = json.load(f)
    return plates_data

json_file = 'indicePlacas.json'

plates_data = load_plates_from_json(json_file)

image_files = os.listdir(folder_path)


for filename in image_files:
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)

        result = reader.readtext(image_path)

        print(f"Texto Lido na Imagem: {filename}")

        detected_text = result[0][1] if result else ""  # Obtém o texto detectado pela primeira região de texto
        detected_text_normalized = apply_substitutions(detected_text.upper())  # Aplica substituições comuns no texto detectado
            
        index_str = filename.split('_')[1].split('.')[0]
        
        try:
            index = int(index_str)
        except ValueError:
            print(f"Erro: Não foi possível extrair o índice do arquivo {filename}")
            continue
        
        real_text = None
        for plate in plates_data:  # Procura a placa correspondente no arquivo JSON
            if plate['indice'] == index:
                real_text = plate['placa']
                break
        
        if real_text is None:
            print(f"Placa para o arquivo {filename} não encontrada no JSON.")
            continue

        print(f"Texto Real da Placa: {real_text}")
        print(f"Texto Detectado pelo OCR: {detected_text}")

        real_text_normalized = apply_substitutions(real_text.upper())

        if detected_text_normalized == real_text_normalized:  # Compara os textos normalizados
            print("Resultado CORRETO!")
        else:
            print("Resultado INCORRETO!")
            print("Verifique a placa manualmente.")

        img.close()