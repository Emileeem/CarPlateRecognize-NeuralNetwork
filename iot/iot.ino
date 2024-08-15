#include <SPI.h>
#include <Ethernet.h>
#include <ArduinoJson.h>
#include <Servo.h>

byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED }; 
const char* host = "http://127.0.0.1:5000/"; 

EthernetClient client;

Servo meuservo;
int angulo = 0;
String lastPlate = ""; 
int lastExists = 0;    

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("Conectando à rede Ethernet...");
  Ethernet.begin(mac);
  delay(1000); 

  meuservo.attach(9);
  meuservo.write(0); 
  lastPlate = "";
  lastExists = 0;
}

void loop() {
  if (client.connect(host, 80)) {
    client.println("GET /dados.json HTTP/1.1");
    client.println("Host: api.exemplo.com"); 
    client.println("Connection: close");
    client.println();

    String response = "";
    while (client.available()) {
      response += (char)client.read();
    }
    client.stop();

    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, response);

    if (error) {
      Serial.print("Falha ao analisar JSON: ");
      Serial.println(error.c_str());
      delay(10000); 
      return;
    }

    int exists = doc["exists"];
    const char* plate = doc["plate"];

    Serial.print("Exists: ");
    Serial.println(exists);
    Serial.print("Plate: ");
    if (plate == nullptr) {
      Serial.println("null");
    } else {
      Serial.println(plate);
    }

    if (String(plate) != lastPlate && exists == 1) {
      angulo = 180;  
      delay(8000);
      angulo = 0;
    } else {
      angulo = 0;    
    }

    meuservo.write(angulo);

    lastPlate = plate ? String(plate) : "";
    lastExists = exists;
  } else {
    Serial.println("Falha na conexão");
  }

  delay(10000);
}
