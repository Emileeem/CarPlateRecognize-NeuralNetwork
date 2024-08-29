import os
import cv2
import easyocr
import mysql.connector
import ssl
import time
from datetime import datetime
import threading
import warnings
import firebase_admin
from firebase_admin import credentials, db

warnings.filterwarnings("ignore", category=FutureWarning)

ssl._create_default_https_context = ssl._create_unverified_context

# Firebase conectando
cred = credentials.Certificate('other/iot-tcc-ef560-firebase-adminsdk-9zvl6-7091569d94.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iot-tcc-ef560-default-rtdb.firebaseio.com/'
})

# Banco de dados
db_conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="CarPlateDB"
)

cursor = db_conn.cursor()

haarcascade = "model/haarcascade_russian_plate_number.xml"
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

min_area = 500  
reader = easyocr.Reader(['pt'], verbose=False)

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
    '2': 'Z',
    'Z': '2',
    "'": ""
}

def apply_substitutions(text):
    for key, value in substitution_map.items():
        text = text.replace(key, value)
    return text

def check_plate_in_database(plate):
    query = "SELECT * FROM Carro WHERE Placa = %s"
    cursor.execute(query, (plate,))
    result = cursor.fetchone()
    return result is not None

def insert_log(carro_id, funcionario_id, hora_entrada, data_entrada):
    query = "INSERT INTO Log (CarroID, FuncionarioID, HoraEntrada, DiaEntrada) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (carro_id, funcionario_id, hora_entrada, data_entrada))
    db_conn.commit()

def get_car_and_func_info(plate):
    query = """
    SELECT Carro.ID AS CarroID, Funcionario.ID AS FuncionarioID
    FROM Carro
    INNER JOIN Funcionario ON Carro.FuncionarioID = Funcionario.ID
    WHERE Carro.Placa = %s
    """
    cursor.execute(query, (plate,))
    return cursor.fetchone()  # Retorna (carro_id, funcionario_id) ou None

def send_to_firebase(plate, exists):
    ref = db.reference('plates')
    ref.set({
        'plate': plate,
        'exists': exists,
        'timestamp': datetime.now().isoformat()
    })

last_detected_plate = None
last_detection_time = time.time()

def detect_plate():
    global last_detected_plate, last_detection_time
    while True:
        success, img = cap.read()
        
        if time.time() - last_detection_time >= 2:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            plate_cascade = cv2.CascadeClassifier(haarcascade)  

            plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)  

            for (x, y, w, h) in plates:
                area = w * h
                if area > min_area:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (127, 187, 200), 2)
                    cv2.putText(img, "Numero da Placa", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                    y_offset = int(h * 0.3) 
                    img_roi = img[y + y_offset: y + h, x: x + w]
                    result = reader.readtext(img_roi)

                    detected_text = result[-1][1] if result else ""  
                    detected_text = apply_substitutions(detected_text.upper()) 
                    
                    print(f"Placa detectada: {detected_text}")

                    if(last_detected_plate == detected_text):
                        print("Placa já lida, aguardando proximo veiculo")
                    else:
                        if len(detected_text) == 7 and detected_text[3].isdigit(): 
                            last_detected_plate = detected_text
                            car_info = get_car_and_func_info(detected_text)
                            
                            if car_info:
                                carro_id, funcionario_id = car_info
                                if check_plate_in_database(detected_text):
                                    print("Placa encontrada no banco de dados!")
                                    current_time = datetime.now().time()
                                    current_date = datetime.now().date()
                                    insert_log(carro_id, funcionario_id, current_time, current_date)
                                    print(f"Log inserido: CarroID={carro_id}, FuncionarioID={funcionario_id}, HoraEntrada={current_time}, DiaEntrada={current_date}")
                                    send_to_firebase(detected_text, 1)
                                else:
                                    print("Placa NAO encontrada no banco de dados!")
                                    send_to_firebase(detected_text, 0)
                            else:
                                print("Nenhuma informação encontrada para a placa.")
                                send_to_firebase(detected_text, 0)
                        else:
                            print("Placa invalida, tentando novamente...")
                        
                    cv2.imshow("img corte", img_roi)

            last_detection_time = time.time()
            
        cv2.imshow("Resultado", img)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    cursor.close()
    db_conn.close()

if __name__ == "__main__":
    detect_plate()
