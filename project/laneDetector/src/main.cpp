/*!
 * @file main.cpp
 * @author Roberto J Baquerizo (roba8460@colorado.edu)
 * @brief
 * @version 1.0
 * @date 2022-08-05
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "lanedetector.h"
#include "thread_utils.h"
#include "thread.h"
#include "configuration.h"

#include "Logger.h"
#include "Sequencer.h"

#include <stdio.h>
#include <time.h>
#include <memory>
#include <pthread.h>
#include <fcntl.h>
#include "unistd.h"

pthread_mutex_t grayscaleBufferLock;
pthread_mutex_t framesProcessedLock;
pthread_mutex_t rawBufferLock;

bool abortS1;
bool abortS2;
bool abortS3;
bool abortS4;
bool abortSequencer;

sem_t* semS1;
sem_t* semS2;
sem_t* semS3;
sem_t* semS4;

uint64_t framesProcessed = 0;

static void createSemaphoresAndMutexes()
{
    pthread_mutex_init( &rawBufferLock, NULL );
    pthread_mutex_init( &grayscaleBufferLock, NULL );
    pthread_mutex_init( &framesProcessedLock, NULL );

    semS1 = sem_open( SEMS1_NAME, O_CREAT | O_EXCL, 0644, 0 );
    if( semS1 == SEM_FAILED )
    {
        sem_unlink( SEMS1_NAME );
        semS1 = sem_open( SEMS1_NAME, O_CREAT | O_EXCL, 0644, 0 );
        if( semS1 == SEM_FAILED )
        {
            perror( "Failed to initialize S1 semaphore" );
            exit( -1 );
        }
    }

    semS2 = sem_open( SEMS2_NAME, O_CREAT | O_EXCL, 0644, 0 );
    if( semS2 == SEM_FAILED )
    {
        sem_unlink( SEMS2_NAME );
        semS2 = sem_open( SEMS2_NAME, O_CREAT | O_EXCL, 0644, 0 );
        if( semS2 == SEM_FAILED )
        {
            perror( "Failed to initialize S2 semaphore" );
            exit( -1 );
        }
    }

    semS3 = sem_open( SEMS3_NAME, O_CREAT | O_EXCL, 0644, 0 );
    if( semS3 == SEM_FAILED )
    {
        sem_unlink( SEMS3_NAME );
        semS3 = sem_open( SEMS3_NAME, O_CREAT | O_EXCL, 0644, 0 );
        if( semS3 == SEM_FAILED )
        {
            perror( "Failed to initialize S3 semaphore" );
            exit( -1 );
        }
    }
    semS4 = sem_open( SEMS4_NAME, O_CREAT | O_EXCL, 0644, 0 );
    if( semS4 == SEM_FAILED )
    {
        sem_unlink( SEMS4_NAME );
        semS4 = sem_open( SEMS4_NAME, O_CREAT | O_EXCL, 0644, 0 );
        if( semS4 == SEM_FAILED )
        {
            perror( "Failed to initialize S4 semaphore" );
            exit( -1 );
        }
    }
}

static const char* keys = {
    "{help          h|false|show help message}"
    "{@input         |./videos/22400003.AVI|path to test video to process}"
    "{store          |./outputFrames/|path to create processed output video}"
    "{show           |false|show intermediate steps}"
};

void printHelp( cv::CommandLineParser* p_parser )
{
    printf( "The program uses Hough line detection and Haar feature detection to identify and mark\n"
            "road lanes and cars on a video stream input\n\r" );
    p_parser->printMessage();
    printf( "Press the ESC key to exit the program.\n\r" );
}

#define USEC_PER_MSEC ( 1000 )
#define SEC_TO_MSEC ( 1000 )
#define NSEC_PER_SEC ( 1000000000 )
#define NSEC_PER_USEC ( 1000000 )


int main( int argc, char** argv )
{
    cv::CommandLineParser* p_parser = new cv::CommandLineParser( argc, argv, keys );
    if( p_parser->get<bool>( "help" ) )
    {
        printHelp( p_parser );
        exit( 0 );
    }

    bool show = p_parser->get<bool>( "show" );
    bool doStore = p_parser->has( "store" );
    cv::String store = p_parser->get<cv::String>( "store" );
    cv::String videoInput = p_parser->get<cv::String>( "@input" );

    pthread_mutex_lock( &framesProcessedLock );
    framesProcessed = 0;
    pthread_mutex_unlock( &framesProcessedLock );
    if( videoInput.empty() )
    {
        LogFatal( "Missing path to test video!" );
        printHelp( p_parser );
        exit( -1 );
    }
    else
    {
        LogInfo( "Using %s as source", videoInput.c_str() );
    }

    double sequencerDeadline = 1 / ( double )Sequencer::SEQUENCER_FREQUENCY;

    createSemaphoresAndMutexes();

    Sequencer* p_sequencer = new Sequencer();
    if( p_sequencer == NULL )
    {
        LogFatal( "Sequencer creation failed!" );
        exit( -1 );
    }
    p_sequencer->setDeadline( sequencerDeadline );

    LineDetector* p_detector = new LineDetector( threadConfigurations,
                                                 LineDetector::DEFAULT_DEVICE_ID,
                                                 videoInput,
                                                 doStore,
                                                 store,
                                                 show );

    if( p_detector == NULL )
    {
        LogFatal( "Detector creation failed!" );
        exit( -1 );
    }
    if( not p_detector->createdOk() )
    {
        delete p_detector;
        exit( -1 );
    }

    char winInput;

    struct timespec start = { 0, 0 };
    struct timespec stop = { 0, 0 };

    LogInfo( "Creating semaphores and mutexes!" );
    clock_gettime( CLOCK_REALTIME, &start );

    while( true )
    {
        p_detector->updateDisplayWindow();

        winInput = cv::waitKey( 10 );
        if( winInput == 27 )
        {
            LogInfo( "User requested exit!" );
            break;
        }
        if( not p_detector->isAlive() )
        {
            LogInfo( "Detector shut down!" );
            break;
        }
        if( p_detector->isDone () )
        {
            break;
        }
    }

    if( p_sequencer )
    {
        LogTrace( "Shutting down Sequencer..." );
        abortSequencer = true;
        while( p_sequencer->isAlive() )
        {
            LogTrace( "Waiting for Sequencer to abort..." );
            sleep( 1 );
            continue;
        }
        delete p_sequencer;
        p_sequencer = NULL;
    }

    if( p_detector )
    {
        p_detector->printFrameRates();

        if( p_detector->isAlive() )
        {
            sem_post( semS1 );
            sem_post( semS2 );
            sem_post( semS3 );
            sem_post( semS4 );

            abortS1 = true;
            abortS2 = true;
            abortS3 = true;
            abortS4 = true;
            while( p_detector->isAlive() )
            {
                LogTrace( "Waiting for threads to abort..." );
                p_detector->shutdown();
                sleep( 1 );
                continue;
            }
            delete p_detector;
            p_detector = NULL;
        }
    }

    if( p_parser )
    {
        delete p_parser;
        p_parser = NULL;
    }

    clock_gettime( CLOCK_REALTIME, &stop );
    double deltas = delta_t( &stop, &start );
    double deltaTMS = deltas / framesProcessed;
    double deltaT = deltaTMS / 1000.0;

    printf( "**** DISPLAY UPDATE RATE ****\n\r" );
    printf( "Frames Processed: %ld\n\r", framesProcessed );
    printf( "Average Display Frame Rate: %3.2f ms per frame\n\r", deltaTMS );
    printf( "Average Display Frame Rate: %3.2f frames per sec (fps)\n\r", 1.0 / deltaT );
    printf( "**** DISPLAY UPDATE RATE ****\n\r" );

    sem_close( semS1 );
    sem_close( semS2 );
    sem_close( semS3 );
    sem_close( semS4 );

    sem_unlink( SEMS1_NAME );
    sem_unlink( SEMS2_NAME );
    sem_unlink( SEMS3_NAME );
    sem_unlink( SEMS4_NAME );

    pthread_mutex_unlock( &grayscaleBufferLock );
    pthread_mutex_destroy( &grayscaleBufferLock );
    pthread_mutex_unlock( &framesProcessedLock );
    pthread_mutex_destroy( &framesProcessedLock );

    return 0;
}
