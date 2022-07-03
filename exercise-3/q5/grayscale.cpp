/**
 * @file grayscale.cpp
 * @brief This program converts the input video to grayscale images saved as PGMs
 */
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    CommandLineParser parser( argc, argv,
                              "{@input | ../Dark-Room-Laser-Spot.mpeg | input video}" );

    VideoCapture capture( parser.get<String>( "@input" ) );

    if( !capture.isOpened() )
    {
        printf( "Coult not open input video stream!\n\r" );
        exit( -1 );
    }

    Mat frame;
    int framesProcessed = 0;

    Mat bgr[3];

    char winInput;

    String currentFrame = "Current Frame";

    namedWindow( currentFrame, WINDOW_NORMAL );
    resizeWindow( currentFrame, Size( 860, 480 ) );

    char filename[100];
    vector<int> compression_params;
    compression_params.push_back( IMWRITE_PXM_BINARY );
    compression_params.push_back( 1 );
    while( true )
    {
        capture.read( frame );

        if( frame.empty() )
        {
            break;
        }

        Mat new_image = Mat::zeros( frame.size(), frame.type() );

        for( int y = 0; y < frame.rows; y++ )
        {
            for( int x = 0; x < frame.cols; x++ )
            {

                new_image.at<Vec3b>( y, x )[0] = 0;
                new_image.at<Vec3b>( y, x )[1] = saturate_cast<uchar>( frame.at<Vec3b>( y, x )[1] );
                new_image.at<Vec3b>( y, x )[2] = 0;
            }
        }
        sprintf( filename, "./output/frame%04d_out.pgm", framesProcessed );
        split( new_image, bgr );
        imwrite( filename, bgr[1], compression_params );

        imshow( currentFrame, bgr[1] );
        framesProcessed++;

        winInput = waitKey( 1 );
        if( 27 == winInput )
        {
            break;
        }
    }

    capture.release();

    destroyWindow( currentFrame );

    return 0;
}
