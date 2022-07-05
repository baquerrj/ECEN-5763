/**
 * @file houghclines.cpp
 * @brief This program demonstrates line finding with the Hough transform
 */
#include <stdio.h>
#include <time.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

const char* sourceImage = "Source";
const char* standardHough = "Detected Lines (in red) - Standard Hough Line Transform";
const char* probabilisticHough = "Detected Lines (in red) - Probabilistic Line Transform";
int min_threshold = 50;
int max_trackbar = 150;

int s_trackbar = max_trackbar;

int hl_threshold = min_threshold + s_trackbar;

void updateThreshold( int newValue, void* object )
{
    s_trackbar = newValue;
    hl_threshold = min_threshold + s_trackbar;
}

double delta_t( struct timespec* stop, struct timespec* start )
{
    double current = ( (double)stop->tv_sec * 1000.0 ) +
        ( (double)( (double)stop->tv_nsec / 1000000.0 ) );
    double last = ( (double)start->tv_sec * 1000.0 ) +
        ( (double)( (double)start->tv_nsec / 1000000.0 ) );
    return ( current - last );
}

void applyHoughlines( Mat* src, Mat* dst, Mat* cdst, Mat* cdstP )
{
    //![edge_detection]
       // Edge detection
    Canny( *src, *dst, 50, 200, 3 );
    //![edge_detection]

    // Copy edges to the images that will display the results in BGR
    cvtColor( *dst, *cdst, COLOR_GRAY2BGR );
    *cdstP = cdst->clone();

    //![hough_lines]
    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines( *dst, lines, 1, CV_PI / 180, hl_threshold, 0, 0 ); // runs the actual detection
    //![hough_lines]
    //![draw_lines]
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos( theta ), b = sin( theta );
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound( x0 + 1000 * ( -b ) );
        pt1.y = cvRound( y0 + 1000 * ( a ) );
        pt2.x = cvRound( x0 - 1000 * ( -b ) );
        pt2.y = cvRound( y0 - 1000 * ( a ) );
        line( *cdst, pt1, pt2, Scalar( 0, 0, 255 ), 3, LINE_AA );
    }
    //![draw_lines]

    //![hough_lines_p]
    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP( *dst, linesP, 1, CV_PI / 180, 50, 50, 10 ); // runs the actual detection
    //![hough_lines_p]
    //![draw_lines_p]
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        line( *cdstP, Point( l[0], l[1] ), Point( l[2], l[3] ), Scalar( 0, 0, 255 ), 3, LINE_AA );
    }
    //![draw_lines_p]

    return;
}

int main( int argc, char** argv )
{
    CommandLineParser parser( argc, argv,
                              "{camera        c|false|Use camera as source}"
                              "{video         v|false|Use video file as source}"
                              "{@videoInput    |../Dark-Room-Laser-Spot.mpeg|video source}"
                              "{help    h|false|show help message}" );
    bool help = parser.get<bool>( "help" );
    if( help )
    {
        printf( "The sample uses Sobel or Scharr OpenCV functions for edge detection\n\n" );
        parser.printMessage();
        printf("\nPress 'ESC' to exit program.\nPress 'R' to reset values ( ksize will be -1 equal to Scharr function )");
        return 0;
    }

    bool useCamera = parser.get<bool>( "camera" );
    bool useVideo = parser.get<bool>( "video" );

    String videoInput = parser.get<String>( "@videoInput" );

    // Declare the output variables
    Mat dst, cdst, cdstP;

    VideoCapture capture;
    if( useCamera )
    {
        printf( "Using camera as source" );
        if( not capture.open( 0 ) )
        {
            printf( "Could not open /dev/video0 as source!" );
            return -1;
        }
        if( !capture.isOpened() )
        {
            printf( "Could not open /dev/video0 as source!" );
            exit( -1 );
        }

        capture.set( CAP_PROP_FRAME_WIDTH, 640 );
        capture.set( CAP_PROP_FRAME_HEIGHT, 480 );
    }
    else if( useVideo )
    {
        printf( "Using video as source" );
        if( not capture.open( videoInput ) )
        {
            printf( "Could not open %s as source!\n\r", videoInput.c_str() );
            return -1;
        }
    }

    namedWindow( sourceImage );
    namedWindow( standardHough );
    namedWindow( probabilisticHough );

    createTrackbar( "Standard HL Threshold:", standardHough, &s_trackbar, max_trackbar, updateThreshold, NULL );

    char winInput;

    Mat src;
    // used to calculate fps in while-loop
    struct timespec start = { 0, 0 };
    struct timespec stop = { 0, 0 };
    double deltas = 0.0;

    int framesProcessed = 0;

    while( true )
    {
        capture.read( src );
        clock_gettime( CLOCK_REALTIME, &start );
        applyHoughlines( &src, &dst, &cdst, &cdstP );
        clock_gettime( CLOCK_REALTIME, &stop );
        deltas += delta_t( &stop, &start );
        framesProcessed++;

        //![imshow]
        // Show results
        imshow( sourceImage, src );
        imshow( standardHough, cdst );
        imshow( probabilisticHough, cdstP );
        //![imshow]
        if( ( winInput = waitKey( 10 ) ) == 27 )
        {
            break;
        }
    }

    double deltaTMS = deltas / framesProcessed;
    double deltaT = deltaTMS / 1000.0;
    printf( "Average Frame Rate: %3.2f ms per frame\n\r", deltaTMS );
    printf( "Average Frame Rate: %3.2f frames per sec (fps)\n\r", 1.0 / deltaT );

    return 0;
}
