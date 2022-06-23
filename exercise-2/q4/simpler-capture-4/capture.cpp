/*
 *
 *  Example by Sam Siewert
 *
 *  Created for OpenCV 4.x for Jetson Nano 2g, based upon
 *  https://docs.opencv.org/4.1.1
 *
 *  Tested with JetPack 4.6 which installs OpenCV 4.1.1
 *  (https://developer.nvidia.com/embedded/jetpack)
 *
 *  Based upon earlier simpler-capture examples created
 *  for OpenCV 2.x and 3.x (C and C++ mixed API) which show
 *  how to use OpenCV instead of lower level V4L2 API for the
 *  Linux UVC driver.
 *
 *  Verify your hardware and OS configuration with:
 *  1) lsusb
 *  2) ls -l /dev/video*
 *  3) dmesg | grep UVC
 *
 *  Note that OpenCV 4.x only supports the C++ API
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <syslog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

// See www.asciitable.com
#define ESCAPE_KEY (27)
#define SYSTEM_ERROR (-1)

double delta_t( struct timespec* stop, struct timespec* start )
{
    double current = ( (double)stop->tv_sec * 1000.0 ) +
        ( (double)( (double)stop->tv_nsec / 1000000.0 ) );
    double last = ( (double)start->tv_sec * 1000.0 ) +
        ( (double)( (double)start->tv_nsec / 1000000.0 ) );
    return ( current - last );
}

int main()
{
    // used to calculate total runtime
    struct timespec mainStart = { 0, 0 };
    struct timespec mainStop = { 0, 0 };
    // used to calculate fps in while-loop
    struct timespec start = { 0, 0 };
    struct timespec stop = { 0, 0 };
    double deltas = 0.0;
    int framesProcessed = 0;
    clock_gettime(CLOCK_REALTIME, &mainStart);
    VideoCapture cam0( 0 );
    namedWindow( "video_display" );
    char winInput;

    if( !cam0.isOpened() )
    {
        exit( SYSTEM_ERROR );
    }

    cam0.set( CAP_PROP_FRAME_WIDTH, 640 );
    cam0.set( CAP_PROP_FRAME_HEIGHT, 480 );

    while( framesProcessed < 2000 )
    {
        framesProcessed++;
        clock_gettime( CLOCK_REALTIME, &start );
        Mat frame;
        cam0.read( frame );
        imshow( "video_display", frame );
        clock_gettime( CLOCK_REALTIME, &stop );
        deltas += delta_t( &stop, &start );

        if( ( winInput = waitKey( 10 ) ) == ESCAPE_KEY )
        {
            break;
        }
        else if( winInput == 'n' )
        {
            cout << "input " << winInput << " ignored" << endl;
        }

    }

    destroyWindow( "video_display" );

    double deltaT = deltas / framesProcessed;
    deltaT = deltaT / 1000.0;
    printf( "Average Frame Rate: %3.4f sec/frame\n\r", deltaT );
    printf( "Average Frame Rate: %3.4f frames/sec (fps)\n\r", 1.0 / deltaT );

    syslog( LOG_CRIT, "Average Frame Rate: %3.4f sec/frame\n\r", deltaT );
    syslog( LOG_CRIT, "Average Frame Rate: %3.4f frames/sec (fps)\n\r", 1.0 / deltaT );
};
