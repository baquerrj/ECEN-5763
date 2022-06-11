/**!
 *
 * @file crosshair.cpp
 *
 * @brief This program opens a window with an image in it. The image is edited to
 * have a border and a crosshair drawn at the center of the image
 *
 * @author Roberto Baquerizo
 *
 * References: https://github.com/siewertsmooc/EMVIA-ECEE-5763/tree/main/computer_vision_cv4_tested/simpler-capture-4
 * and https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
 */


#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>


#include "opencv2/core/core.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
 //#include "opencv2/contrib/contrib.hpp"

using namespace cv;
using namespace std;

#define VRES_ROWS (240)
#define HRES_COLS (320)

#define ESC_KEY (27)

int main( int argc, char** argv )
{
    int hres = HRES_COLS;
    int vres = VRES_ROWS;
    int frameCnt = 0;
    int borderSize = 4;
    int markerSize = 25;
    int thickness = 1;
    int lineType = 8;

    // interactive computer vision loop
    namedWindow( "Output Image", WINDOW_AUTOSIZE );

    VideoCapture cam0( 0 );
    if( !cam0.isOpened() )
    {
        exit( -1 );
    }

    cam0.set( CAP_PROP_FRAME_WIDTH, 320 );
    cam0.set( CAP_PROP_FRAME_HEIGHT, 240 );

    Mat frame;
    cam0.read( frame );

    cv::drawMarker( frame, Point( hres / 2, vres / 2 ), Scalar( 0, 255, 255 ),
                    MARKER_CROSS, markerSize, thickness, lineType ); // draw crosshair at center of frame

    copyMakeBorder( frame, frame, borderSize, borderSize,
                    borderSize, borderSize, BORDER_CONSTANT,
                    Scalar( 100, 255, 255 ) ); // Make the border of the frame

    if( !frame.data )  // Check for invalid input
    {
        printf( "Error capturing an image!\n" );
        exit( -1 );
    }

    // Display the image and exit when ESC key is pressed
    while( 1 )
    {
        frameCnt++;

        imshow( "Output Image", frame );

        imwrite( "output.png", frame );
        char c = waitKey( 10 );

        if( c == ESC_KEY )
        {
            exit( 1 );
        }

    }

    return 0;
}
