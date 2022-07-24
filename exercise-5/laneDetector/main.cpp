
#include "lanedetector.hpp"

#include <stdio.h>
#include <time.h>
#include <memory>

static const char* keys = {
    "{help          h|false|show help message}"
    "{@input         |./videos/GP010639.MP4|path to test video to process}"
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

    LineDetector* detector = new LineDetector( 0, videoInput );

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

        winInput = waitKey( 2 );
        if( winInput == 27 )
        {
            break;
        }
    }

    if( detector != NULL )
    {
        delete detector;
    }

    return 0;
}
