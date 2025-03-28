//Make sure to install the SD library in Arduino IDE's library manager

#include "SPI.h"
#include "SD.h"//http://librarymanager/All#SD

//THE PROGRAM WILL PRINT THE DETECTED OBJECT DISTANCE IN CM TO THE SERIAL PORT
const byte trigPin = WB_IO1; //Pin numberof sensor trigger pin (transmit)
const byte echoPin = WB_IO2; //Pin number of sensor echo pin (receiver)

//CONSTANTS
#define CM_PER_MICROSECOND .0343 //Speed of sound in cm per microsecond
#define TRIP_DISTANCE 400 //Distance to trip sensor in cm
#define UNDEFINED_DISTANCE 1000 //placeholder used to label distances recorded outside of sensor range
#define FILE_NAME "RAK_DATA.txt" //name of .txt file created in SD card by writeFile
#define DATA_STRING_LENGTH 15 
#define DATA_ARRAY_SIZE 100

//ALGO CONSTANTS(values are currently somewhat arbitrary)
#define CAR_HEIGHT_MIN 1.4  // Car height range in meters
#define CAR_HEIGHT_MAX 2.0  
#define CAR_TIME_THRESHOLD 1.5  // Maximum time for a car to pass
#define HEIGHT_CHANGE_THRESHOLD 0.2  // Minimum change to consider motion

//Variables

int counter = 0;
int array_index = 0;
long duration; //Time between ultrasound trasmit and receive
int distance; //Distance from sensor to object
int distances[10];
char data_string[DATA_STRING_LENGTH]; //used in writeFileArray, DONT KNOW
char data_array[DATA_ARRAY_SIZE][DATA_STRING_LENGTH]; //used in writeFileArray, DONT KNOW

//ADDED FOR ALGO
int delta_distance = 0;
int min_height = 0;
int previous_height = 0;
bool object_detected = false;
long start_time;
long elapsed_time;
int car_detected = 0;



//

void setup() {
  Serial.begin(115200); //Serial.begin(speed); sets data rate to speed bits per second. data rate is limited by how good hardware is?
  delay(1000); //delays for 1000 ms

  pinMode(trigPin, OUTPUT); //sets output pin
  pinMode(echoPin, INPUT); //sets input pin
  distances[0] = 0; //sets first value of distances array to 0, likely done to prevent stupid errors. could pose problem for splitting?

  if (!SD.begin()) { // Adjust WB_IO3 to match your SD module CS pin
    Serial.println("SD Card initialization failed!");
    return;
  }
  Serial.println("SD Card initialized successfully.");

  deleteFile(FILE_NAME);
}

void loop() {
  delay(40);
  digitalWrite(trigPin, LOW); //sets trigPin OFF. Likely for calibration purposes, however possibly uneeded? check why its here, likely for hardware limitation reasons
  delayMicroseconds(10);
 
  //Send a 10 microsecond ultrasonic pulse
  digitalWrite(trigPin, HIGH); //sets trigPin ON
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW); //sets trigPin OFF
  //Measure microseconds between pulse trasmission and reception
  duration = pulseIn(echoPin, HIGH); 
//below tree edge checks for far or really far distances, or if the sensor just fucks up for some reason. establishes the "fucked up" distance as UNDEFINED_DISTANCE (300)
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
  
  //ALGO WORK
  delta_distance = distance - previous_height;
  
  if (abs(delta_distance) > HEIGHT_CHANGE_THRESHOLD && object_detected == false)
  {
    object_detected = true;
    start_time = millis();
    min_height = distance;

  }
  else if (object_detected == true)
  {
    min_height = MIN(min_height, distance);
    
    if (abs(delta_distance) < HEIGHT_CHANGE_THRESHOLD)
    {
      elapsed_time = millis() - start_time;

      if (CAR_HEIGHT_MIN <= min_height && min_height <= CAR_HEIGHT_MAX && elapsed_time <= CAR_TIME_THRESHOLD)
      {
        car_detected = 1;
      }
      object_detected = false;
      min_height = distance;
    }
  }
  previous_height = distance;




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


//writeFileArray: Writes to SD card

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