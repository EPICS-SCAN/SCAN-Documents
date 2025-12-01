#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define MAX_DATA_POINTS 10000

// Function declarations
bool isCar(float data[][2], int n);
int getObjectCount(const char* dataFile); 
int readObjectFromPython(const char* dataFile, int objectIndex, float data[][2]);

int main() {
    const char* dataFile = "RAK_DATA_SPLIT_F2025_Test2.txt"; //data file path

    // Get total number of objects
    int totalObjects = getObjectCount(dataFile); 
    if (totalObjects <= 0) {
        fprintf(stderr, "Error: No objects found or failed to read file\n");
        return 1;
    }

    printf("Total objects to analyze: %d\n\n", totalObjects);

    // Allocate results array
    bool* isCarArray = (bool*)malloc(totalObjects * sizeof(bool));
    if (!isCarArray) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Allocate memory for object data
    float (*data)[2] = malloc(MAX_DATA_POINTS * sizeof(*data));
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        free(isCarArray);
        return 1;
    }

    int carCount = 0;

    for (int i = 0; i < totalObjects; i++) {
        printf("=== Analyzing Object %d ===\n", i);

        int dataSize = readObjectFromPython(dataFile, i, data);
        if (dataSize < 0) {
            fprintf(stderr, "Error reading object %d\n", i);
            isCarArray[i] = false;
            continue;
        }

        printf("Data points: %d\n", dataSize);

        // Normalize time to start at 0
        float t0 = data[0][0];
        for (int j = 0; j < dataSize; j++) {
            data[j][0] -= t0;
        }

        // Run robust car detection
        bool carDetected = isCar(data, dataSize);

        if (carDetected) {
            printf("✓ Object %d: IS A CAR\n", i);
            isCarArray[i] = true;
            carCount++;
        } else {
            printf("✗ Object %d: NOT A CAR\n", i);
            isCarArray[i] = false;
        }

        printf("\n");
    }

    // Summary
    printf("====== SUMMARY ======\n");
    printf("Total objects analyzed: %d\n", totalObjects);
    printf("Cars detected: %d\n", carCount);
    printf("Non-cars: %d\n\n", totalObjects - carCount);

    printf("Results array:\n");
    for (int i = 0; i < totalObjects; i++) {
        printf("Object %d: %s\n", i, isCarArray[i] ? "CAR" : "NOT CAR");
    }

    free(data);
    free(isCarArray);

    return 0;
}


bool isCar(float data[][2], int n)
{
    if (n < 10) return false;

    // Smooth distances (3-pt moving average)
    float dist[n];
    for (int i = 0; i < n; i++) {
        float d0 = data[i][1];
        float d1 = (i > 0) ? data[i-1][1] : d0;
        float d2 = (i < n-1) ? data[i+1][1] : d0;
        dist[i] = (d0 + d1 + d2) / 3.0f;
    }

    // Compute ground baseline (median first 10)
    float first[10];
    int k = (n < 10 ? n : 10);
    for (int i = 0; i < k; i++) first[i] = dist[i];

    for (int i = 0; i < k - 1; i++)
        for (int j = i + 1; j < k; j++)
            if (first[j] < first[i]) {
                float tmp = first[i]; first[i] = first[j]; first[j] = tmp;
            }

    float ground = first[k/2];

    // Find the actual minimum
    float minDist = dist[0];
    int minIdx = 0;
    for (int i = 1; i < n; i++) {
        if (dist[i] < minDist) {
            minDist = dist[i];
            minIdx = i;
        }
    }

    float dip = ground - minDist;
    if (dip < 12) {
        printf("  Car test failed: dip too small (%.1f)\n", dip);
        return false;
    }

    // Track valley width around minimum
    int left = minIdx;
    while (left > 0 && dist[left - 1] <= dist[left] + 10)
        left--;

    int right = minIdx;
    while (right < n - 1 && dist[right + 1] <= dist[right] + 10)
        right++;

    float duration_ms = data[right][0] - data[left][0];

    if (duration_ms < 80) {
        printf("  Car test failed: dip width too small (%f ms)\n", duration_ms);
        return false;
    }

    // Slope check
    float enterSlope = fabs(dist[left] - dist[left + 2]);
    float exitSlope  = fabs(dist[right - 2] - dist[right]);

    if (enterSlope < 2 || exitSlope < 2) {
        printf("  Car test failed: slope too weak (%.1f, %.1f)\n",
               enterSlope, exitSlope);
        return false;
    }

    printf("  Car OK: dip %.1f, width %.1f ms\n", dip, duration_ms);
    return true;
}


// Helper functions 
int getObjectCount(const char* dataFile) {
    char command[512];
    snprintf(command, sizeof(command), "python sensor_reader.py \"%s\" count", dataFile);

    FILE* pipe = _popen(command, "r");
    if (!pipe) {
        fprintf(stderr, "Failed to run python script\n");
        return -1;
    }

    int count;
    if (fscanf(pipe, "%d", &count) != 1) {
        _pclose(pipe);
        return -1;
    }

    _pclose(pipe);
    return count;
}

int readObjectFromPython(const char* dataFile, int objectIndex, float data[][2]) {
    char command[512];
    snprintf(command, sizeof(command), "python sensor_reader.py \"%s\" %d", dataFile, objectIndex);

    FILE* pipe = _popen(command, "r");
    if (!pipe) {
        fprintf(stderr, "Failed to run python script\n");
        return -1;
    }

    int size;
    if (fscanf(pipe, "%d", &size) != 1 || size < 0) {
        _pclose(pipe);
        return -1;
    }

    for (int i = 0; i < size; i++) {
        if (fscanf(pipe, "%f,%f", &data[i][0], &data[i][1]) != 2) {
            fprintf(stderr, "Error reading data point %d\n", i);
            _pclose(pipe);
            return -1;
        }
    }

    _pclose(pipe);
    return size;
}
