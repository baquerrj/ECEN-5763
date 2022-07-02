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

typedef struct {
    double x_left;
    double x_right;
    double y_top;
    double y_bottom;
} edges_t;

class Tracker
{

public:
    const String WINDOW_NAME = "Video Display";
public:
    Tracker( const String directory )
    {
        myDirectory = directory;
        myEdges = { 0, 0, 0, 0 };
    }
    ~Tracker() {}

    void loadNextImage();
    void threshold();
    void rasterImage();
    void findCenter();
    void drawCrosshairs();
    void saveImage();
    inline bool isImageEmpty()
    {
        return myImage.empty();
    }

    inline Mat* getImage()
    {
        return &myImage;
    }

private:
    String myDirectory;
    Mat myImage;
    double x_bar;
    double y_bar;
    int myFrameCounter;
    edges_t myEdges;
};

void Tracker::threshold()
{
    for( int y = 0; y < myImage.rows; y++ )
    {
        for( int x = 0; x < myImage.cols; x++ )
        {
            if( myImage.at<uchar>( y, x ) > 40 )
            {
                myImage.at<uchar>( y, x ) = 255;
            }
            else
            {
                myImage.at<uchar>( y, x ) = 0;
            }
        }
    }
}

void Tracker::rasterImage()
{
    medianBlur( myImage, myImage, 3 );
    bool foundFirstEdge = false;
    for( int y = 0; y < myImage.rows; y++ )
    {
        for( int x = 0; x < myImage.cols; x++ )
        {
            if( not foundFirstEdge )
            {
                if( myImage.at<uchar>( y, x ) == 255 )
                {
                    if( ( myImage.at<uchar>( y - 1, x - 1 ) == 0 ) and
                        ( myImage.at<uchar>( y - 1, x ) == 0 ) and
                        ( myImage.at<uchar>( y, x - 1 ) == 0 ) )
                    {
                        foundFirstEdge = true;
                        printf( "Found the first edge at (x,y) = (%d,%d)\n\r", x, y );
                        myEdges.x_left = x;
                        myEdges.y_top = y;
                    }
                }
            }
            else
            {
                if( myImage.at<uchar>( y, x ) == 0 )
                {
                    if( ( myImage.at<uchar>( y - 1, x - 1 ) == 255 ) and
                        ( myImage.at<uchar>( y - 1, x ) == 255 ) and
                        ( myImage.at<uchar>( y, x - 1 ) == 255 ) )
                    {
                        printf( "\nlast edge at x:%d y:%d", x, y );
                        myEdges.x_right = x;
                        myEdges.y_bottom = y;
                        return;
                    }
                }
            }
        }
    }
}

void Tracker::findCenter()
{
    x_bar = ( myEdges.x_right - myEdges.x_left ) / 2 + myEdges.x_left;
    y_bar = ( myEdges.y_bottom - myEdges.y_top ) / 2 + myEdges.y_top;
}

void Tracker::drawCrosshairs()
{
    char text[50];
    printf( "\nx_bar:%f y_bar:%f", x_bar, y_bar );
    sprintf( text, "x_bar:%f y_bar:%f", x_bar, y_bar );
    line( myImage, Point( myEdges.x_left - 30, y_bar ), Point( myEdges.x_right + 30, y_bar ), 255, 3, 8 );
    line( myImage, Point( x_bar, myEdges.y_top - 30 ), Point( x_bar, myEdges.y_bottom + 30 ), 255, 3, 8 );
    putText( myImage, text, Point( 30, 30 ), FONT_HERSHEY_COMPLEX_SMALL, 0.8, 255, 1, LINE_AA );
    imshow( "image", myImage );
}


void Tracker::loadNextImage()
{
    char filename[100];
    sprintf( filename, "frame%04d_out.pgm", myFrameCounter );
    printf( "filename: %s\n\r", ( myDirectory + filename ).c_str() );
    myImage = imread( ( myDirectory + filename ), IMREAD_GRAYSCALE );
    myFrameCounter++;
}

void Tracker::saveImage()
{
    char filename[100];
    sprintf( filename, "./output/frame%04d.pgm", myFrameCounter );
    imwrite( filename, myImage );
}

int main( int argc, char** argv )
{

    CommandLineParser parser( argc, argv,
                              "{@input | ../q5/output/ | input video}" );

    String directory = parser.get<String>( "@input" );
    char winInput;

    String currentFrame = "Current Frame";

    Tracker tracker( directory );
    Mat* image;

    while( true )
    {
        tracker.loadNextImage();
        if( tracker.isImageEmpty() )
        {
            break;
        }

        tracker.threshold();

        tracker.rasterImage();

        tracker.findCenter();
        tracker.drawCrosshairs();

        tracker.saveImage();

        image = tracker.getImage();

        imshow( currentFrame, *image );

        winInput = waitKey( 2 );
        if( 27 == winInput )
        {
            break;
        }
    }

    return 0;
}
