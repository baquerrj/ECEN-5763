#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include <iostream>

using namespace cv;
using namespace std;


Mat applySkeletal( Mat* src )
{
    Mat gray, binary, mfblur;

    cvtColor( *src, gray, COLOR_BGR2GRAY );

    // Use 70 negative for Moose, 150 positive for hand
    //
    // To improve, compute a histogram here and set threshold to first peak
    //
    // For now, histogram analysis was done with GIMP
    //
    threshold( gray, binary, 70, 255, THRESH_BINARY );
    binary = 255 - binary;

    // To remove median filter, just replace blurr value with 1
    medianBlur( binary, mfblur, 1 );

    // This section of code was adapted from the following post, which was
    // based in turn on the Wikipedia description of a morphological skeleton
    //
    // http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
    //
    Mat skel( mfblur.size(), CV_8UC1, Scalar( 0 ) );
    Mat temp;
    Mat eroded;
    Mat element = getStructuringElement( MORPH_CROSS, Size( 3, 3 ) );
    bool done;
    int iterations = 0;

    do
    {
        erode( mfblur, eroded, element );
        dilate( eroded, temp, element );
        subtract( mfblur, temp, temp );
        bitwise_or( skel, temp, skel );
        eroded.copyTo( mfblur );

        done = ( countNonZero( mfblur ) == 0 );
        iterations++;

    } while( !done && ( iterations < 100 ) );

    cout << "iterations=" << iterations << endl;

    return skel;
}


int main()
{
    VideoCapture camera( 0 );
    Mat src;

    int framesProcessed = 0;
    char winInput;

    while( framesProcessed < 3000 )
    {
        camera.read( src );

        if( src.empty() )
        {
            cout << "could not read from camera" << endl;
            continue;;
        }

        // show original source image and wait for input to next step
        imshow( "source", src );
        Mat skel = applySkeletal( &src );

        imshow( "skeleton", skel );

        winInput = waitKey( 10 );
        if( 'q' == winInput )
        {
            break;
        }
        framesProcessed++;
    }

    return 0;
}
