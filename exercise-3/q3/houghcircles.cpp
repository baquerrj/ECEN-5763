/**
 * @file houghcircles.cpp
 * @brief This program demonstrates circle finding with the Hough transform
 */
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

void applyHoughCircles( Mat * src )
{
    //![convert_to_gray]
    Mat gray;
    cvtColor( *src, gray, COLOR_BGR2GRAY );
    //![convert_to_gray]

    //![reduce_noise]
    medianBlur( gray, gray, 5 );
    //![reduce_noise]

    //![houghcircles]
    vector<Vec3f> circles;
    HoughCircles( gray, circles, HOUGH_GRADIENT, 1,
                  gray.rows / 16,  // change this value to detect circles with different distances to each other
                  100, 30, 1, 30 // change the last two parameters
                  // (min_radius & max_radius) to detect larger circles
    );
    //![houghcircles]

    //![draw]
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
        Point center = Point( c[0], c[1] );
        // circle center
        circle( *src, center, 1, Scalar( 0, 100, 100 ), 3, LINE_AA );
        // circle outline
        int radius = c[2];
        circle( *src, center, radius, Scalar( 255, 0, 255 ), 3, LINE_AA );
    }
    //![draw]
}

int main(int argc, char** argv)
{
    VideoCapture cam0( 0 );

    if( !cam0.isOpened() )
    {
        exit( -1 );
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
        applyHoughCircles( &src );
        clock_gettime( CLOCK_REALTIME, &stop );
        deltas += delta_t( &stop, &start );
        framesProcessed++;
        imshow( "detected circles", src );

        if( ( winInput = waitKey( 10 ) ) == 27 )
        {
            break;
        }
    }
    //![display]
    waitKey();
    //![display]

    return 0;
}
