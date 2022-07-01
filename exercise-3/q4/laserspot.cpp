/**
 * @file laserspot.cpp
 * @brief This program performs background elimination and preserves the moving laser dot
 */
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    VideoWriter video;

    Mat prev, curr, diff;

    int framesProcessed = 0;

    VideoCapture capture( "../Dark-Room-Laser-Spot-with-Clutter.mpeg" );

    Size frameSize = Size( (int)capture.get( CAP_PROP_FRAME_WIDTH ), (int)capture.get( CAP_PROP_FRAME_HEIGHT ) );

    video.open( "processed.mp4", VideoWriter::fourcc( 'M', 'P', '4', 'V' ),
                capture.get( CAP_PROP_FPS ), frameSize, true );

    if( !video.isOpened() )
    {
        printf( "Could not open output video stream!\n\r" );
        exit( -1 );
    }

    if( !capture.isOpened() )
    {
        printf( "Coult not open input video stream!\n\r" );
        exit( -1 );
    }

    capture.read( prev );

    int numberOfChannels = prev.channels();
    Mat prevBGR[numberOfChannels];
    Mat currBGR[numberOfChannels];
    Mat diffBGR[numberOfChannels];

    char winInput;

    String previousFrame = "Previous Frame";
    String currentFrame = "Current Frame";
    String diffFrame = "Diff Frame";

    namedWindow( previousFrame, WINDOW_GUI_NORMAL );
    namedWindow( currentFrame, WINDOW_GUI_NORMAL );
    namedWindow( diffFrame, WINDOW_GUI_NORMAL );

    while( true )
    {
        split( prev, prevBGR );
        capture.read( curr );

        if( curr.empty() )
        {
            break;
        }

        split( curr, currBGR );

        for( int i = 0; i < 3; i++ )
        {
            absdiff( prevBGR[i], currBGR[i], diffBGR[i] );
        }

        prev = curr;

        merge( diffBGR, numberOfChannels, diff );

        framesProcessed++;

        video.write( diff );

        imshow( previousFrame, prev );
        imshow( currentFrame, curr );
        imshow( diffFrame, diff );

        winInput = waitKey( 2 );
        if( 27 == winInput )
        {
            break;
        }
    }

    capture.release();
    video.release();

    destroyWindow( previousFrame );
    destroyWindow( currentFrame );
    destroyWindow( diffFrame );

    return 0;
}
