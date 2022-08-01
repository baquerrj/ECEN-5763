
#include "lanedetector.hpp"
#include "thread_utils.hpp"
#include "thread.hpp"
#include "configuration.hpp"

#include "Logger.h"

#include <stdio.h>
#include <time.h>
#include <memory>
#include <pthread.h>
#include <fcntl.h>
#include "unistd.h"

pthread_mutex_t grayscaleBufferLock;
pthread_mutex_t imageLock;
pthread_mutex_t rawBufferLock;

bool abortS1;
bool abortS2;
bool abortS3;
bool abortS4;

sem_t* semS1;
sem_t* semS2;
sem_t* semS3;
sem_t* semS4;

static void createSemaphoresAndMutexes()
{
    pthread_mutex_init( &rawBufferLock, NULL );
    pthread_mutex_init( &grayscaleBufferLock, NULL );
    pthread_mutex_init( &imageLock, NULL );

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
            perror( "Failed to initialize S3 semaphore" );
            exit( -1 );
        }
    }
}

double delta_t( struct timespec* stop, struct timespec* start )
{
    double current = ( ( double )stop->tv_sec * 1000.0 ) +
        ( ( double )( ( double )stop->tv_nsec / 1000000.0 ) );
    double last = ( ( double )start->tv_sec * 1000.0 ) +
        ( ( double )( ( double )start->tv_nsec / 1000000.0 ) );
    return ( current - last );
}

static const char* keys = {
    "{help          h|false|show help message}"
    "{@input         |./videos/22400003.AVI|path to test video to process}"
    "{store          ||path to create processed output video}"
    "{show           |false|show intermediate steps}"
};

void printHelp( CommandLineParser* p_parser )
{
    printf( "The program uses Hough line detection and Haar feature detection to identify and mark\n"
            "road lanes and cars on a video stream input\n\r" );
    p_parser->printMessage();
    printf( "Press the ESC key to exit the program.\n\r" );
}

int main( int argc, char** argv )
{
    CommandLineParser* p_parser = new CommandLineParser( argc, argv, keys );
    if( p_parser->get<bool>( "help" ) )
    {
        printHelp( p_parser );
        exit( 0 );
    }

    bool show = p_parser->get<bool>( "show" );
    bool doStore = p_parser->has( "store" );
    String store = p_parser->get<String>( "store" );
    String videoInput = p_parser->get<String>( "@input" );

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
    // pthread_mutex_init( &grayscaleBufferLock, NULL );
    // pthread_mutex_init( &imageLock, NULL );

    createSemaphoresAndMutexes();
    LineDetector* p_detector = new LineDetector( threadConfigurations,
                                                 LineDetector::DEFAULT_DEVICE_ID,
                                                 videoInput,
                                                 doStore,
                                                 store );

    abortS3 = true;
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

    int framesProcessed = 0;
    struct timespec start = { 0, 0 };
    struct timespec stop = { 0, 0 };

    LogInfo( "Creating semaphores and mutexes!" );
    clock_gettime( CLOCK_REALTIME, &start );

    while( true )
    {
        // if( not p_detector->newFrameReady() )
        // {
        //     continue;
        // }

        // if( p_detector->isFrameEmpty() )
        // {
        //     break;
        // }
        // p_detector->prepareImage();
        // p_detector->detectLanes();
        // p_detector->detectCars();


            p_detector->showLanesImage();

        if( doStore )
        {
            p_detector->writeFrameToVideo();
        }

        winInput = waitKey( 2 );
        if( winInput == 27 )
        {
            break;
        }
    }
    if( p_detector )
    {
        if( p_detector->isAlive() )
        {
            abortS1 = true;
            abortS2 = true;
            abortS3 = true;
            while( p_detector->isAlive() )
            {
                // loop here until all threads shut down
                continue;
            }
            framesProcessed = p_detector->getFramesProcessed();
            delete p_detector;
            p_detector = NULL;
        }
    }


    clock_gettime( CLOCK_REALTIME, &stop );
    double deltas = delta_t( &stop, &start );
    double deltaTMS = deltas / framesProcessed;
    double deltaT = deltaTMS / 1000.0;


    printf( "Average Frame Rate: %3.2f ms per frame\n\r", deltaTMS );
    printf( "Average Frame Rate: %3.2f frames per sec (fps)\n\r", 1.0 / deltaT );

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
    pthread_mutex_unlock( &imageLock );
    pthread_mutex_destroy( &imageLock );

    return 0;
}
