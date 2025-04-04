//Make sure to install the SD library in Arduino IDE's library manager

#include "SPI.h"
#include "SD.h"//http://librarymanager/All#SD
#include <vector>

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




//Variables
std::vector<int> time_vector;
std::vector<int> distance_vector;
int array_index = 0;
long duration; //Time between ultrasound trasmit and receive
int distance; //Distance from sensor to object
char data_string[DATA_STRING_LENGTH]; //used in writeFileArray, DONT KNOW
char data_array[DATA_ARRAY_SIZE][DATA_STRING_LENGTH]; //used in writeFileArray, DONT KNOW

//ADDED FOR ALGO
int prev_distance1 = 0;
int prev_distance2 = 0;
int prev_distance3 = 0;
int curr_distance = 0;
int prev_time1 = 0;
int prev_time2 = 0;
int prev_time3 = 0;
int curr_time  = 0;
bool high = true;
int end_count = 0;



void setup() {
  Serial.begin(115200); //Serial.begin(speed); sets data rate to speed bits per second. data rate is limited by how good hardware is?
  delay(1000); //delays for 1000 ms

  pinMode(trigPin, OUTPUT); //sets output pin
  pinMode(echoPin, INPUT); //sets input pin

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


  //ALGO WORK
  //format (time, distance)
  prev_time1 = prev_time2;
  prev_time2 = prev_time3;
  prev_time3 = curr_time;
  curr_time = millis();
  prev_distance1 = prev_distance2;
  prev_distance2 = prev_distance3;
  prev_distance3 = curr_distance;

NEW SKETCH

  if (distance < 190 && prev_time1 != 0 && high)
  {
  //NOTE: other possible option is to create 2 dimensional vector... I don't know how the efficiency compares for either option
    time_vector.push_back(prev_time1);
    time_vector.push_back(prev_time2);
    time_vector.push_back(prev_time3);
    time_vector.push_back(curr_time);
    distance_vector.push_back(prev_distance1);
    distance_vector.push_back(prev_distance2);
    distance_vector.push_back(prev_distance3);
    distance_vector.push_back(distance);
    high = false;
    end_count = 0;
  }
  else if (distance < 190 && !(high))
  {
    end_count = 0;
    time_vector.push_back(curr_time);
    distance_vector.push_back(distance);
  }
  else if (end_count == 3)
  {
  //end chunk of data
    carDetectionAlgo(time_vector, distance_vector);
    time_vector.clear();
    distance_vector.clear();
    high = true;
    end_count = 0;
  }
  else if (distance > 190 && !(high))
  {
    end_count = end_count + 1;
    time_vector.push_back(curr_time);
    distance_vector.push_back(distance);  
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

int carDetectionAlgo(const std::vector<int>& time_vector, const std::vector<int>& distance_vector)
{
  return 0;
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