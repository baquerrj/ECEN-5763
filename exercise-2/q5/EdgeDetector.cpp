/**
 * @file EdgeDector.cpp
 * @brief Program that toggles between Sobel and Canny for edge detection
 * @author Roberto Baquerizo
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static int ksize = 1;
static int scale = 1;
static int delta = 0;
static int ddepth = CV_16S;

const String windowName = "Edge Detection Demo";

int lowThreshold = 0;
const int max_lowThreshold = 100;
const int cannyRatio = 3;
const int kernel_size = 3;

const int maxKsize = 31;


class EdgeDetector
{
public:
    enum detector_e
    {
        CANNY,
        SOBEL,
        NONE
    };

public:
    EdgeDetector( int deviceId = 0, int frameWidth = 640, int frameHeight = 480 )
    {
        myCamera = VideoCapture( deviceId );
        if( !myCamera.isOpened() )
        {
            printf( "Error opening camera %d\n\r", deviceId );
            exit( -1 );
        }
        myCamera.set( CAP_PROP_FRAME_WIDTH, frameWidth );
        myCamera.set( CAP_PROP_FRAME_HEIGHT, frameHeight );

        currentDetector = NONE;

        myKsize = ksize;
        myScale = scale;
        myDelta = delta;

        namedWindow( myWindowName );

        createTrackbars();

    }
    ~EdgeDetector() {}

    inline void readCameraFrame()
    {
        myCamera.read( mySource );
    }

    inline void setCurrentDetector( detector_e detection )
    {
        currentDetector = detection;
    }

    inline void showImage()
    {
        imshow( myWindowName, myImageToShow );
    }

    inline void createTrackbars()
    {
        createTrackbar( "Canny Threshold:", myWindowName, &lowThreshold, max_lowThreshold, updateThreshold, this );
        createTrackbar( "Sobel Kernel Size", myWindowName, &ksize, maxKsize, updateKsize, this );
        createTrackbar( "Sobel Scale", myWindowName, &scale, 100, updateScale, this );
        createTrackbar( "Sobel Delta", myWindowName, &delta, 100, updateDelta, this );
    }

    inline static void updateThreshold( int newValue, void* object )
    {
        EdgeDetector* ed = (EdgeDetector*)object;

        ed->setCannyThreshold( newValue );
    }

    inline static void updateKsize( int newValue, void* object )
    {
        EdgeDetector* ed = (EdgeDetector*)object;

        ed->setKsize( newValue );
    }

    inline static void updateScale( int newValue, void* object )
    {
        EdgeDetector* ed = (EdgeDetector*)object;

        ed->setScale( newValue );
    }

    inline static void updateDelta( int newValue, void* object )
    {
        EdgeDetector* ed = (EdgeDetector*)object;

        ed->setDelta( newValue );
    }

    inline void setCannyThreshold( int value )
    {
        myThreshold = value;
    }

    inline void setKsize( int value )
    {
        if( value % 2 == 0 )
        {
            myKsize = value + 1;
            setTrackbarPos( "ksize", myWindowName, myKsize );
        }
        else
        {
            myKsize = value;
        }
    }
    inline void setScale( int value )
    {
        myScale = value;
    }
    inline void setDelta( int value )
    {
        myDelta = value;
    }

    void applyCanny();

    void applySobel();

    inline void applyTransformation()
    {
        switch( currentDetector )
        {
            case CANNY:
                applyCanny();
                myImageToShow = myDestination;
                break;
            case SOBEL:
                applySobel();
                myImageToShow = myDestination;
                break;
            case NONE:
            default:
                myImageToShow = mySource;
                break;
        }
    }

    inline int openImage( String name )
    {
        myWindowName = name;
        mySource = imread( name, IMREAD_COLOR );
        if( mySource.empty() )
        {
            printf( "Error opening image: %s\n", myWindowName.c_str() );
            return 1;
        }
        else
        {
            return 0;
        }
    }

private:
    detector_e currentDetector;
    Mat mySource;
    Mat myDestination;
    Mat detected_edges;
    Mat myImageToShow;
    String myWindowName;
    VideoCapture myCamera;
    int myThreshold;
    int myKsize;
    int myScale;
    int myDelta;
};

void EdgeDetector::applyCanny()
{
    Mat src_gray;
    cvtColor( mySource, src_gray, COLOR_BGR2GRAY );

    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size( 3, 3 ) );

    /// Canny detector
    Canny( detected_edges, detected_edges, myThreshold, myThreshold * cannyRatio, kernel_size );

    /// Using Canny's output as a mask, we display our result
    myDestination = Scalar::all( 0 );

    mySource.copyTo( myDestination, detected_edges );
}

void EdgeDetector::applySobel()
{
    Mat src_gray;
    Mat gray;

    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur( mySource, gray, Size( 3, 3 ), 0, 0, BORDER_DEFAULT );

    // Convert the image to grayscale
    cvtColor( mySource, src_gray, COLOR_BGR2GRAY );

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Sobel( src_gray, grad_x, ddepth, 1, 0, myKsize, myScale, myDelta, BORDER_DEFAULT );

    /// Gradient Y
    Sobel( src_gray, grad_y, ddepth, 0, 1, myKsize, myScale, myDelta, BORDER_DEFAULT );

    // converting back to CV_8U
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, myDestination );
}

int main( int argc, char** argv )
{
    cv::CommandLineParser parser( argc, argv,
                                  "{help    h|false|show help message}" );
    bool help = parser.get<bool>( "help" );
    if( help )
    {
        cout << "The sample uses Sobel or Canny OpenCV functions for edge detection\n\n";
        parser.printMessage();
        cout << "Press 'ESC' to exit program.\nPress 'c' to apply Canny edge detection\n";
        cout << "Press 's' to apply Sobel edge detection\n";
        cout << "Press 'n' to disable edge detection\n";
        return 0;
    }

    EdgeDetector * edgeDetector = new EdgeDetector();

    edgeDetector->readCameraFrame();
    edgeDetector->applyTransformation();
    edgeDetector->showImage();

    while( true )
    {
        edgeDetector->readCameraFrame();
        edgeDetector->applyTransformation();
        edgeDetector->showImage();

        char key = (char)waitKey( 1 );

        if( key == 27 )
        {
            break;
        }
        else if( key == 'C' || key == 'c' )
        {
            edgeDetector->setCurrentDetector( EdgeDetector::CANNY );
        }
        else if( key == 'S' || key == 's' )
        {
            edgeDetector->setCurrentDetector( EdgeDetector::SOBEL );
        }
        else if( key == 'N' || key == 'n' )
        {
            edgeDetector->setCurrentDetector( EdgeDetector::NONE );
        }
    }

    if( edgeDetector )
    {
        delete edgeDetector;
    }
    return 0;
}
