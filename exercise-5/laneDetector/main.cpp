
#include "lanedetector.hpp"

#include <stdio.h>
#include <time.h>
#include <memory>

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
    CommandLineParser parser( argc, argv, keys );
    if( parser.get<bool>( "help" ) )
    {
        printHelp( &parser );
        exit( 0 );
    }

    bool show = parser.get<bool>( "show" );
    bool doStore = parser.has( "store" );
    String store = parser.get<String>( "store" );
    String videoInput = parser.get<String>( "@input" );

    if( videoInput.empty() )
    {
        printf( "Missing path to test video!\n\r" );
        printHelp( &parser );
        exit( -1 );
    }
    else
    {
        printf( "Using %s as source\n\r", videoInput.c_str() );
    }

    LineDetector* detector = new LineDetector( LineDetector::DEFAULT_DEVICE_ID,
                                                videoInput,
                                                doStore,
                                                store );


    if( detector == NULL )
    {
        exit( -1 );
    }

    if( not detector->loadClassifier( LineDetector::CAR_CLASSIFIER ) )
    {
        delete detector;
        exit( -1 );
    }

    char winInput;

    int framesProcessed = 0;
    struct timespec start = { 0, 0 };
    struct timespec stop = { 0, 0 };
    clock_gettime( CLOCK_REALTIME, &start );

    while( true )
    {
        detector->readFrame();
        if( detector->isFrameEmpty() )
        {
            break;
        }
        detector->prepareImage();
        detector->detectLanes();
        detector->detectCars();

        framesProcessed++;

        if( show )
        {
            detector->showLanesImage();
            detector->showVehiclesImage();
            detector->showSourceImage();
        }
        else
        {
            detector->showLanesImage();
        }

        if( doStore )
        {
            detector->writeFrameToVideo();
        }

        winInput = waitKey( 2 );
        if( winInput == 27 )
        {
            break;
        }
    }
    clock_gettime( CLOCK_REALTIME, &stop );
    double deltas = delta_t( &stop, &start );
    double deltaTMS = deltas / framesProcessed;
    double deltaT = deltaTMS / 1000.0;
    printf( "Average Frame Rate: %3.2f ms per frame\n\r", deltaTMS );
    printf( "Average Frame Rate: %3.2f frames per sec (fps)\n\r", 1.0 / deltaT );

    if( detector != NULL )
    {
        delete detector;
    }

    // video.release();
    return 0;
}
