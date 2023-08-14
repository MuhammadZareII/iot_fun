


#include <ESP8266WiFi.h>


const char* ssid = "ssid";
const char* password = "pass";
 
int LED = 2;                 // led connected to D0
WiFiServer server(80);
 
void setup() 
{
  Serial.begin(115200);
  pinMode(LED, OUTPUT);
  digitalWrite(LED, HIGH);
 
  Serial.print("Connecting to WiFi ");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) 
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");
 
 /*-------- server started---------*/ 
  server.begin();
  Serial.println("Server started");
 
  /*------printing ip address--------*/
  Serial.print("IP Address of network: ");
  Serial.println(WiFi.localIP());
  Serial.print("Copy and paste the following URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/");
}
 
void loop() 
  {
    WiFiClient client = server.available();    
    if (!client) 
    {
      return;
    }
 // Serial.println("Waiting for new client");   
  while(!client.available())
  {
    delay(1);
  }
 
  String request = client.readStringUntil('\r');
  Serial.println(request);
  client.flush();
 
 
  int value = 0;
  if (request.indexOf("/LED=ON") != -1)  
  {
    digitalWrite(LED, HIGH);
    value = 1;
  }
  else if (request.indexOf("/LED=OFF") != -1)  
  {
    digitalWrite(LED, LOW);
    value = 0;
  }
  else if (request.indexOf("/DATA") != -1)  
  {
    value = -1;
  }
 
/*------------------Creating html page---------------------*/

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/html");
  client.println(""); 
  client.println("<!DOCTYPE HTML>");
  client.println("<html>");
 
 
  if(value == 1) 
  {
    client.print("LED is: ON");
  } 
  if(value == 0) 
  {
    client.print("LED is: OFF");
  }

  client.println("<br><br>");
  client.println("<a href=\"/LED=ON\"\"><button>LED ON</button></a>");
  client.println("<a href=\"/LED=OFF\"\"><button>LED OFF</button></a><br />");  
  //client.println("<a href=\"/DATA\"\"><button>READ DATA</button></a><br />"); 
  client.println("</html>");
 
 // delay(1);
 // Serial.println("Client disonnected");
 // Serial.println("");
 
}

