/*!
 * @file configuration.h
 * @author Roberto J Baquerizo (roba8460@colorado.edu)
 * @brief
 * @version 1.0
 * @date 2022-08-05
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __CONFIGURATION_H__
#define __CONFIGURATION_H__

#include <thread_utils.h>
#include <semaphore.h>
#include "Logger.h"


/**
 * @brief calculates the difference between the start and stop times
 *
 * @param stop
 * @param start
 * @return double
 */
inline double delta_t( struct timespec* stop, struct timespec* start )
{
    double current = ( ( double )stop->tv_sec * 1000.0 ) +
        ( ( double )( ( double )stop->tv_nsec / 1000000.0 ) );
    double last = ( ( double )start->tv_sec * 1000.0 ) +
        ( ( double )( ( double )start->tv_nsec / 1000000.0 ) );
    return ( current - last );
}


//! @file Defines thread and process configuration stuff

#define SEMS1_NAME "/SEMS1"
#define SEMS2_NAME "/SEMS2"
#define SEMS3_NAME "/SEMS3"
#define SEMS4_NAME "/SEMS4"

extern bool abortS1;
extern bool abortS2;
extern bool abortS3;
extern bool abortS4;
extern sem_t* semS1;
extern sem_t* semS2;
extern sem_t* semS3;
extern sem_t* semS4;

// Thread CPU affinities. (negative value = no affinity specified)
const int NUM_CPUS = 4;  // number of CPU's on the target machine

const int CPU_MAIN = 0;
const int CPU_SEQUENCER = CPU_MAIN;
const int CPU_CAPTURE = CPU_MAIN;
const int CPU_LANE_DETECTION = 1;
const int CPU_CAR_DETECTION = 2;
const int CPU_SIGN_DETECTION = 3;
const int CPU_ANNOTATION = 3;

static const ProcessParams sequencerParams = {
    CPU_SEQUENCER,
    SCHED_FIFO,
    99,
    0 };

static const ThreadConfigData sequencerThreadConfig = {
    true,
    "sequencer",
    sequencerParams };

static const ProcessParams captureParams = {
    CPU_CAPTURE,
    SCHED_FIFO,
    98,
    0};

static const ThreadConfigData captureThreadConfig = {
    true,
    "capture",
    captureParams};

static const ProcessParams laneDetectionParams = {
    CPU_LANE_DETECTION,
    SCHED_FIFO,
    99,
    0};

static const ThreadConfigData laneDetectionThreadConfig = {
    true,
    "laneDetection",
    laneDetectionParams};

static const ProcessParams carDetectionParams = {
    CPU_CAR_DETECTION,  // CPU1
    SCHED_FIFO,
    99,  // highest priority
    0};

static const ThreadConfigData carDetectionThreadConfig = {
    true,
    "carDetection",
    carDetectionParams};

static const ProcessParams annotationParams = {
    CPU_ANNOTATION,
    SCHED_FIFO,
    99,
    0 };

static const ThreadConfigData annotationThreadConfig = {
    true,
    "annotation",
    annotationParams };

static const ThreadConfigData threadConfigurations[ 4 ] ={
    captureThreadConfig,
    laneDetectionThreadConfig,
    carDetectionThreadConfig,
    annotationThreadConfig
};

extern bool abortS1;
extern bool abortS2;
extern bool abortS3;
extern bool abortS4;
extern bool abortSequencer;
extern sem_t* semS1;
extern sem_t* semS2;
extern sem_t* semS3;
extern sem_t* semS4;

extern uint64_t framesProcessed;

#endif // __CONFIGURATION_H__