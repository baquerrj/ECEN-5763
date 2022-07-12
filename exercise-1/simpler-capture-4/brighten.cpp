// Written by Sam Siewert and XXX
//
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <time.h>

using namespace cv; using namespace std;
double alpha=1.0;  int beta=10;  /* contrast and brightness control */

double delta_t( struct timespec* stop, struct timespec* start )
{
    double current = ( (double)stop->tv_sec * 1000.0 ) +
        ( (double)( (double)stop->tv_nsec / 1000000.0 ) );
    double last = ( (double)start->tv_sec * 1000.0 ) +
        ( (double)( (double)start->tv_nsec / 1000000.0 ) );
    return ( current - last );
}

int main( int argc, char** argv )
{
    struct timespec start;
    struct timespec stop;
    int framesProcessed = 1;
    double deltas = 0.0;
    // Mat is a matrix object
    Mat image = imread( argv[1] ); // read in image file

    Mat new_image = Mat::zeros( image.size(), image.type() );

    // Check command line arguments
    if(argc < 2)
    {
	    printf("Usage: brighten <input-file>\n");
            exit(-1);
    }


    std::cout<<"* Enter alpha brighten factor [1.0-3.0]: ";std::cin>>alpha;
    std::cout<<"* Enter beta contrast increase value [0-100]: "; std::cin>>beta;
    clock_gettime( CLOCK_REALTIME, &start );

    // Do the operation new_image(i,j) = alpha*image(i,j) + beta
    for( int y = 0; y < image.rows; y++ )
    {
        for( int x = 0; x < image.cols; x++ )
        {
            for( int c = 0; c < 3; c++ )
                new_image.at<Vec3b>(y,x)[c] =
                    saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
        }
    }

    namedWindow("Original Image", 1); namedWindow("New Image", 1);
    imshow("Original Image", image); imshow("New Image", new_image);

    clock_gettime( CLOCK_REALTIME, &stop );
    deltas = delta_t( &stop, &start );
    double deltaTMS = deltas / framesProcessed;
    double deltaT = deltaTMS / 1000.0;
    printf( "Average Frame Rate: %3.2f ms per frame\n\r", deltaTMS );
    printf( "Average Frame Rate: %3.2f frames per sec (fps)\n\r", 1.0 / deltaT );

    imwrite("single_core_sharpened.ppm", new_image );
    waitKey(); return 0;
}
