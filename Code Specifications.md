**Data Splitting**
Take our sensor data as input, and output a cut-down set of data that contains each detected object 'dip' separated by delimiters.

A 'dip' object:

---(delimiter)
Time,distance <
Time,distance < 
Time,distance <  The detected time and distance values corresponding to the object
Time,distance < 
---(delimiter)


Input: .txt file, in format:
Time,distance
Time,distance
Time,distance
Time,distance


Output: .txt file, in format:
Each 

Time,distance
Time,distance
Time,distance
---
Time,distance
Time,distance
---

