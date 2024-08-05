import numpy as np
import cv2
import pytesseract
import mysql.connector

# Configurações do OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ajuste conforme necessário

# Configuração do YOLO
labels_path = './classes.names'
labels = open(labels_path).read().strip().split('\n')
weights_path = './lapi.weights'
configuration_path = './darknet-yolov3.cfg'
probability_minimum = 0.4
threshold = 0.3
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[i[0]-1] for i in network.getUnconnectedOutLayers()]

# Conectar ao banco de dados MySQL
def conectar_banco():
    return mysql.connector.connect(
        host='localhost',
        user='root', 
        password='root',  
        database='CarPlateDB'
    )

# Função para verificar a placa no banco de dados
def verificar_placa(placa):
    conn = conectar_banco()
    cursor = conn.cursor()
    cursor.execute('SELECT Placa FROM Carro WHERE Placa = %s', (placa,))
    resultado = cursor.fetchone()
    conn.close()
    return resultado is not None

# Função para aplicar substituições comuns
def aplicar_substituicoes(texto):
    substituicoes = {
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
    for chave, valor in substituicoes.items():
        texto = texto.replace(chave, valor)
    return texto

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # largura
cap.set(4, 480)  # altura
min_area = 500  # ajuste conforme necessário

while True:
    success, img = cap.read()
    if not success:
        break

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    network.setInput(blob)
    output_from_network = network.forward(layers_names_output)
    
    h, w = img.shape[:2]
    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in output_from_network:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
    
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            img_roi = img[y_min:y_min + box_height, x_min:x_min + box_width]

            texto_detectado = pytesseract.image_to_string(img_roi, config='--psm 8').strip()
            texto_normalizado = aplicar_substituicoes(texto_detectado.upper())

            if verificar_placa(texto_normalizado):
                cv2.putText(img, "Placa Reconhecida!", (x_min, y_min - 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Placa Não Reconhecida", (x_min, y_min - 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            
            cv2.rectangle(img, (x_min, y_min), (x_min + box_width, y_min + box_height), (127, 187, 200), 2)
            cv2.putText(img, texto_detectado, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Resultado", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()