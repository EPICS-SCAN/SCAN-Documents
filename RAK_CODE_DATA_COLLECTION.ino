//Make sure to install the SD library in Arduino IDE's library manager

#include "SPI.h"
#include "SD.h"//http://librarymanager/All#SD

//THE PROGRAM WILL PRINT THE DETECTED OBJECT DISTANCE IN CM TO THE SERIAL PORT
const byte trigPin = WB_IO1; //Pin numberof sensor trigger pin (transmit)
const byte echoPin = WB_IO2; //Pin number of sensor echo pin (receiver)

//CONSTANTS
#define CM_PER_MICROSECOND .0343 //Speed of sound in cm per microsecond
#define TRIP_DISTANCE 400 //Distance to trip sensor in cm
#define UNDEFINED_DISTANCE 1000
#define FILE_NAME "RAK_DATA.txt"
#define DATA_STRING_LENGTH 15
#define DATA_ARRAY_SIZE 100
//Variables

int counter = 0;
int array_index = 0;
long duration; //Time between ultrasound trasmit and receive
int distance; //Distance from sensor to object
int distances[10];
char data_string[DATA_STRING_LENGTH];
char data_array[DATA_ARRAY_SIZE][DATA_STRING_LENGTH];

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  distances[0] = 0;

  if (!SD.begin()) { // Adjust WB_IO3 to match your SD module CS pin
    Serial.println("SD Card initialization failed!");
    return;
  }
  Serial.println("SD Card initialized successfully.");

  deleteFile(FILE_NAME);
}

void loop() {
  delay(40);
  digitalWrite(trigPin, LOW);
  delayMicroseconds(10);
 
  //Send a 10 microsecond ultrasonic pulse
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  //Measure microseconds between pulse trasmission and reception
  duration = pulseIn(echoPin, HIGH);

  if(duration >= 29154){
    distance = UNDEFINED_DISTANCE;
  }
  else if (duration != 0) {
    //Calculate distance
    distance = (duration * CM_PER_MICROSECOND) / 2;
  }
  else {
    //Assume object is very far away
    distance = UNDEFINED_DISTANCE;
  }

  //Add distance to array
  if(distance == UNDEFINED_DISTANCE && counter != 0)
  {
    distances[counter] = distances[counter - 1];
  }
  else if(distance == UNDEFINED_DISTANCE)
  {
    distances[counter] = distances[9];
  }
  else{
    distances[counter] = distance;
  }

  if(counter == 9)
  {
    counter = 0;
  }
  else 
  {
    counter++;
  }

  if(distance == 1000 && counter != 0)
  {
    distance = distances[counter - 1];
  }
  else if(distance == 1000)
  {
    distance = distances[9];
  }
  else{
    distance = distance;
  }
  
  Serial.print(300);
  Serial.print(',');
  Serial.print(distance);
  Serial.print(',');
  Serial.print(0);
  Serial.print('\n');
  

  //sprintf(data_string, "%d,%ld\n", distance, millis());
  //Serial.println(data_string);
  if(array_index < DATA_ARRAY_SIZE - 1)
  {
    sprintf(data_array[array_index], "%d,%ld\n", distance, millis());
    array_index++;
  }
  else
  {
    sprintf(data_array[array_index], "%d,%ld\n", distance, millis());
    writeFileArray(FILE_NAME, data_array);
    array_index = 0;
  }
  
  //writeFile(FILE_NAME, data_string);

 
  //Wait some milliseconds before next cycle
  
}

/*
  Writes message to file at path
  path - path to file 
  message - string to write to file
*/
void writeFile(const char * path, const char * message)
{
  Serial.printf("Writing file: %s\n", path);

  File file = SD.open(path, FILE_WRITE);

  if (file) // if the file opened okay, write to it:
  {
    file.print(message);
    
    file.close();
  }
  else
  {
    Serial.println("Failed to open file for writing.");
    return;
  }
}

void writeFileArray(const char * path, char data_array[DATA_ARRAY_SIZE][DATA_STRING_LENGTH])
{
  Serial.printf("Writing file: %s\n", path);
  unsigned long start = millis();
  File file = SD.open(path, FILE_WRITE);

  if (file) // if the file opened okay, write to it:
  {
    for(int i = 0; i < DATA_ARRAY_SIZE; i++)
    {
      file.print(data_array[i]);
    }
    file.close();
    Serial.printf("Time elapsed: %ld", millis() - start);
  }
  else
  {
    Serial.println("Failed to open file for writing.");
    return;
  }
}

void deleteFile(const char * path)
{
  Serial.printf("Deleting file: %s\n", path);

  if(SD.remove(path))
  {
    Serial.println("File deleted.");
  }
  else
  {
    Serial.println("Delete failed.");
  }
}
