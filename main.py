import os
import cv2
import easyocr
import mysql.connector
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="CarPlateDB"
)

cursor = db.cursor()

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

last_detection_time = time.time()

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

                if len(detected_text) == 7 and detected_text[3].isdigit(): 
                    if check_plate_in_database(detected_text):
                        print("Placa encontrada no banco de dados!")
                    else:
                        print("Placa NÃO encontrada no banco de dados!")
                else:
                    print("Placa inválida, tentando novamente...")
                    
                cv2.imshow("img corte", img_roi)

        last_detection_time = time.time()
        
    cv2.imshow("Resultado", img)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
db.close()
