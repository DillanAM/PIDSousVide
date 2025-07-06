
int HEAT_ELEMENT=6;
int COOLING_VALVE = 7;
int PUMP = 8;
int redPin = 12;
int greenPin = 11;
int bluePin = 10;

void setup()
{
  Serial.begin(9600);
  pinMode(HEAT_ELEMENT, OUTPUT);
  pinMode(COOLING_VALVE, OUTPUT);
  pinMode(PUMP, OUTPUT);
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  
  digitalWrite(HEAT_ELEMENT,0);
  digitalWrite(COOLING_VALVE, 0);
  digitalWrite(PUMP, 1);
  setColor(30, 189, 17);
  Serial.println("Standby Mode");
}

void setColor(int RVAL, int GVAL, int BVAL)
{
  analogWrite(redPin, RVAL);
  analogWrite(greenPin, GVAL);
  analogWrite(bluePin, BVAL);
}

void loop()
{
  char i;
  
  if (Serial.available()){
    i=Serial.read();
    Serial.println("DATA RECEIVED:");
    if(i=='0')
      {
      digitalWrite(HEAT_ELEMENT,0);
      digitalWrite(COOLING_VALVE, 0);
      digitalWrite(PUMP, 1);
      setColor(30, 189, 17);
      Serial.println("Standby Mode");
      }
    if(i=='1')
      {
      digitalWrite(HEAT_ELEMENT,1);
      digitalWrite(COOLING_VALVE, 0);
      digitalWrite(PUMP, 0);
      setColor(255, 20, 0);
      Serial.println("Heating Mode");
      }
    if(i=='2')
      {
      digitalWrite(HEAT_ELEMENT,0);
      digitalWrite(COOLING_VALVE, 1);
      digitalWrite(PUMP, 0);
      setColor(67, 217, 224);
      Serial.println("Cooling Mode");
      }
    if(i=='3')
      {
      digitalWrite(HEAT_ELEMENT,0);
      digitalWrite(COOLING_VALVE, 0);
      digitalWrite(PUMP, 0);
      setColor(255, 200, 0);
      Serial.println("Dwelling Mode");
    }
  }
}