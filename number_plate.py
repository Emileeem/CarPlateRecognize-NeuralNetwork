import os
import cv2

haarcascade = "model/haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)

cap.set(3, 640) # width 
cap.set(4, 480) # height

min_area = 500  # obtendo a média de onde está a placa
count = 0

# Verifica o último número de arquivo na pasta ./placas
placas_folder = './placas/'
existing_files = os.listdir(placas_folder)
if existing_files:
    existing_files.sort()
    last_file = existing_files[-1]
    count = int(last_file.split('_')[1].split('.')[0]) + 1

while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(haarcascade)  # pega o xml para ler
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converte a imagem para a escala de cinza

    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Binariza a imagem


    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)  # busca as coordenadas da placa

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (127, 187, 200), 2)
            cv2.putText(img, "Numero da Placa", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img_bin[y: y + h, x: x + w]
            cv2.imshow("img corte", img_roi)

    cv2.imshow("Resultado", img)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('s'):  # quando pressiona a tecla S
        cv2.imwrite(f"{placas_folder}/placa_{count}.jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Placa Salva", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Resultados", img)
        cv2.waitKey(500)
        count += 1

    elif key & 0xFF == ord('q'):  # quando pressiona a tecla Q
        break

cap.release()
cv2.destroyAllWindows()
