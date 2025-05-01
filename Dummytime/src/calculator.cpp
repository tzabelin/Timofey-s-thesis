#define COMPUTATION_TIME 378
#define SAVE_STEP        10
#define DISK_LATENCY     0 //25255
#define FAILURE_RATE     0.1
#define FAILURE_INTERVAL 378

#define k 3.0
#define N 5000.0

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

int main()
{
    double NS = N / SAVE_STEP;
    double SC = SAVE_STEP * COMPUTATION_TIME;
    double L = DISK_LATENCY;
    double SCF = k * SC / FAILURE_INTERVAL;
    
    double s_term = 1 / (pow(1-FAILURE_RATE, SCF)) - 1;
    double t_term = FAILURE_INTERVAL / (k * FAILURE_RATE) + DISK_LATENCY;

    double results = NS * (SC + L + s_term * t_term);
    
    printf("Result: %.2f\n", results);
    return 0;
}
