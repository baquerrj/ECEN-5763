#ifndef __CONFIGURATION_HPP__
#define __CONFIGURATION_HPP__

#include <thread_utils.hpp>
#include <semaphore.h>
#include "Logger.h"

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
const int CPU_CAPTURE = CPU_MAIN;
const int CPU_LANE_DETECTION = 1;
const int CPU_CAR_DETECTION = 2;
const int CPU_SIGN_DETECTION = 3;
const int CPU_LOGGER = 0;

static const ProcessParams captureParams = {
    cpuCapture,
    SCHED_FIFO,
    99,
    0};

static const ThreadConfigData captureThreadConfig = {
    true,
    "capture",
    captureParams};

static const ProcessParams laneDetectionParams = {
    cpuLaneDetection,
    SCHED_FIFO,
    98,
    0};

static const ThreadConfigData laneDetectionThreadConfig = {
    true,
    "laneDetection",
    laneDetectionParams};

static const ProcessParams carDetectionParams = {
    cpuCarDetection,  // CPU1
    SCHED_FIFO,
    98,  // highest priority
    0};

static const ThreadConfigData carDetectionThreadConfig = {
    true,
    "carDetection",
    carDetectionParams};

static const ProcessParams signDetectionParams = {
    cpuSignDetection,
    SCHED_FIFO,
    98,
    0};

static const ThreadConfigData signDetectionThreadConfig = {
    true,
    "signDetection",
    signDetectionParams};

static const ProcessParams loggerParams = {
    cpuLogger,
    SCHED_FIFO,
    1,  // low priority thread
    0};

static const ThreadConfigData loggerThreadConfig = {
    true,
    "logger",
    loggerParams};

extern bool abortS1;
extern bool abortS2;
extern bool abortS3;
extern bool abortS4;
extern sem_t* semS1;
extern sem_t* semS2;
extern sem_t* semS3;
extern sem_t* semS4;

#endif // __CONFIGURATION_HPP__