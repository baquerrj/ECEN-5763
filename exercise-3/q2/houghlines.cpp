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

double delta_t( struct timespec* stop, struct timespec* start )
{
    double current = ( (double)stop->tv_sec * 1000.0 ) +
        ( (double)( (double)stop->tv_nsec / 1000000.0 ) );
    double last = ( (double)start->tv_sec * 1000.0 ) +
        ( (double)( (double)start->tv_nsec / 1000000.0 ) );
    return ( current - last );
}

void applyHoughlines( Mat* src, Mat* dst, Mat* cdst, Mat* cdstP )
// Mat applyHoughlines( Mat * src, Mat * dst, Mat * cdst )
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
    HoughLines( *dst, lines, 1, CV_PI / 180, 150, 0, 0 ); // runs the actual detection
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

int main(int argc, char** argv)
{
    // Declare the output variables
    Mat dst, cdst, cdstP;

    VideoCapture cam0( 0 );

    if( !cam0.isOpened() )
    {
        exit(-1);
    }

    cam0.set( CAP_PROP_FRAME_WIDTH, 640 );
    cam0.set( CAP_PROP_FRAME_HEIGHT, 480 );

    char winInput;

    Mat src;
    // used to calculate fps in while-loop
    struct timespec start = { 0, 0 };
    struct timespec stop = { 0, 0 };
    double deltas = 0.0;

    int framesProcessed = 0;

    while( true )
    {
        cam0.read( src );
        clock_gettime( CLOCK_REALTIME, &start );
        applyHoughlines( &src, &dst, &cdst , &cdstP );
        clock_gettime( CLOCK_REALTIME, &stop );
        deltas += delta_t( &stop, &start );
        framesProcessed++;

        //![imshow]
        // Show results
        imshow("Source", src);
        imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
        imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
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
