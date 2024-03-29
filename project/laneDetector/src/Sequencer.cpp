/*!
 * @file Sequencer.cpp
 * @author Roberto J Baquerizo (roba8460@colorado.edu)
 * @brief
 * @version 1.0
 * @date 2022-08-05
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <Sequencer.h>
#include <configuration.h>
#include <fcntl.h>
#include <Logger.h>
#include <syslog.h>
#include <thread.h>
#include <thread_utils.h>

#define USEC_PER_MSEC ( 1000 )
#define SEC_TO_MSEC ( 1000 )
#define NSEC_PER_SEC ( 1000000000 )
#define NSEC_PER_USEC ( 1000000 )

#define EXTRA_CYCLES ( 20 )

Sequencer::Sequencer() :
    FrameBase( sequencerThreadConfig )
{
    requiredIterations = ( ( 2000 + EXTRA_CYCLES ) * SEQUENCER_FREQUENCY );
    executionTimes = new double[requiredIterations] {};
    if( executionTimes == NULL )
    {
        LogFatal( "Mem allocation failed for executionTimes for SEQ" );
    }

    startTimes = new double[requiredIterations] {};
    if( startTimes == NULL )
    {
        LogFatal( "Mem allocation failed for startTimes for SEQ" );
    }

    endTimes = new double[requiredIterations] {};
    if( endTimes == NULL )
    {
        LogFatal( "Mem allocation failed for endTimes for SEQ" );
    }

    thread = new CyclicThread( sequencerThreadConfig, Sequencer::execute, this, true );
    if( NULL == thread )
    {
        LogFatal( "Could not allocate memory for SEQ Thread" );
        exit( EXIT_FAILURE );
    }
    alive = true;
}

Sequencer::~Sequencer()
{
    LogTrace( "Entered" );
    LogTrace( "Exiting" );
}

void Sequencer::sequenceServices()
{
    // struct timespec delay_time = { 0, 50000000 };   // delay for 50 msec, 20Hz
    struct timespec delay_time = { 0, 25000000 };   // delay for 25 msec
    // struct timespec delay_time = { 0, 50000000 };   // delay for 50 msec
    struct timespec remaining_time;
    double residual;
    int rc, delay_cnt = 0;

    // 1 Sequencer Period = (FREQUENCY/COUNT) = (50/1) = 50Hz
    // 2 Sequencer Periods = (FREQUENCY/COUNT) = (50/2) = 25Hz
    // 3 Sequencer Periods = (FREQUENCY/COUNT) = (50/3) = 18Hz

    static uint8_t FULL_PERIOD = 1;
    static uint8_t HALF_PERIOD = 2;
    static uint8_t THIRD_PERIOD = 3;
    // static uint8_t divisor = SEQUENCER_FREQUENCY;
    // static uint8_t divisor = SEQUENCER_FREQUENCY / 20;

    do
    {
        if( abortSequencer )
        {
            LogTrace( "Aborting %s.", thread->getName() );
            thread->shutdown();
            return;
        }
        delay_cnt = 0;
        residual = 0.0;

        do
        {
            rc = nanosleep( &delay_time, &remaining_time );

            if( rc == EINTR )
            {
                residual = remaining_time.tv_sec + ( ( double )remaining_time.tv_nsec / ( double )NSEC_PER_SEC );

                if( residual > 0.0 )
                    printf( "residual=%lf, sec=%d, nsec=%d\n", residual, ( int )remaining_time.tv_sec, ( int )remaining_time.tv_nsec );

                delay_cnt++;
            }
            else if( rc < 0 )
            {
                perror( "Sequencer nanosleep" );
                exit( -1 );
            }

        } while( ( residual > 0.0 ) && ( delay_cnt < 100 ) );

        // Calculate Start time
        clock_gettime( CLOCK_REALTIME, &start );

        // Store start time in seconds
        startTimes[ count ] = ( ( double )start.tv_sec + ( double )( ( start.tv_nsec ) / ( double )1000000000 ) );

        syslog( LOG_INFO, "SEQ Count: %llu   Sequencer start Time: %lf seconds\n", count, startTimes[ count ] );

        if( delay_cnt > 1 )
            printf( "Sequencer looping delay %d\n", delay_cnt );

        // Release each service at a sub-rate of the generic sequencer rate
        // Servcie_1 = 25Hz
        if( not abortS1 and ( count % FULL_PERIOD ) == 0 )
        {
            syslog( LOG_INFO, "S1 Release at %llu   Time: %lf seconds\n", count, startTimes[ count ] );
            sem_post( semS1 );
        }

        // Servcie_2 = 25Hz
        if( not abortS2 and ( count % HALF_PERIOD ) == 0 )
        {
            syslog( LOG_INFO, "S2 Release at %llu   Time: %lf seconds\n", count, startTimes[ count ] );
            sem_post( semS2 );
        }

        // Servcie_3 = 25Hz
        if( not abortS3 and ( count % HALF_PERIOD ) == 0 )
        {
            syslog( LOG_INFO, "S3 Release at %llu   Time: %lf seconds\n", count, startTimes[ count ] );
            sem_post( semS3 );
        }

        // Servcie_4 = RT_MIN	@ CAPTURE_FREQUENCY (0.1Hz or 1Hz)
        // if( not abortS4 and ( count % FULL_PERIOD ) == 0 )
        // {
        //     syslog( LOG_INFO, "S4 Release at %llu   Time: %lf seconds\n", count, startTimes[ count ] );
        //     sem_post( semS4 );
        // }


        clock_gettime( CLOCK_REALTIME, &end );
        endTimes[ count ] = ( ( double )end.tv_sec + ( double )( ( end.tv_nsec ) / ( double )1000000000 ) );

        // executionTimes[ count ] = delta_t( &end, &start );

        syslog( LOG_INFO, "SEQ Count: %llu   Sequencer end Time: %lf seconds\n", count, endTimes[ count ] );

        count++;  //Increment the sequencer count
    } while( not abortSequencer and ( count < requiredIterations ) );

    sem_post( semS1 );
    sem_post( semS2 );
    sem_post( semS3 );
    sem_post( semS4 );
}

void* Sequencer::execute( void* context )
{
    ( ( Sequencer* )context )->sequenceServices();
    return NULL;
}