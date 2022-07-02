/**
 * @file commarker.cpp
 * @brief This program performs background elimination and preserves the moving laser dot
 */
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;

class Tracker
{

public:
    const String WINDOW_NAME = "Video Display";
    static const int LINE_THICKNESS = 1;
    static const int LINE_TYPE = 8;
    static const int LINE_SHIFT = 0;

public:
    Tracker( const String filename )
    {
        myFilename = filename;
    }
    ~Tracker();

    void locateXleft( int i, int j );
    void locateXright( int i, int j );
    void locateYtop( int i, int j );
    void locateYbottom( int i, int j );
    void locateCom();

public:
    String myFilename;
    int xleft;
    int xright;
    int xtop;
    int xbottom;
    int yleft;
    int yright;
    int ytop;
    int ybottom;

    int x;
    int y;
};

void Tracker::locateXleft( int i, int j )
{

}
void Tracker::locateXright( int i, int j )
{

}
void Tracker::locateYtop( int i, int j )
{

}
void Tracker::locateYbottom( int i, int j )
{

}
void Tracker::locateCom()
{

}


int main( int argc, char** argv )
{

    CommandLineParser parser( argc, argv,
                              "{@input | ../Dark-Room-Laser-Spot-with-Clutter.mpeg | input video}" );

    VideoCapture capture( parser.get<String>( "@input") );

    Size frameSize = Size( (int)capture.get( CAP_PROP_FRAME_WIDTH ), (int)capture.get( CAP_PROP_FRAME_HEIGHT ) );

    VideoWriter video;
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

    Mat frame;
    capture.read( frame );
    int framesProcessed = 0;

    int numberOfChannels = frame.channels();
    Mat frameBGR[numberOfChannels];

    char winInput;

    String currentFrame = "Current Frame";

    namedWindow( currentFrame, WINDOW_GUI_NORMAL );

    char filename[100];

    while( true )
    {
        capture.read( frame );

        if( frame.empty() )
        {
            break;
        }

        split( frame, frameBGR );

        sprintf(filename, "./output/grayscale_frame_%04d.pgm", framesProcessed );
        imwrite(filename, frameBGR[0]);
        video.write( frameBGR[0] );

        imshow( currentFrame, frameBGR[0] );
        framesProcessed++;

        winInput = waitKey( 2 );
        if( 27 == winInput )
        {
            break;
        }
    }

    capture.release();
    video.release();

    destroyWindow( currentFrame );

    return 0;
}
