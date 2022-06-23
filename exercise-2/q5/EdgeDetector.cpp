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

enum detector_e
{
    CANNY,
    SOBEL,
    NONE
};

class EdgeDetector
{
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

    inline void showImage( bool original = true )
    {
        imshow( myWindowName, myImageToShow );
        // if( original )
        // {
        //     imshow( myWindowName, mySource );
        // }
        // else
        // {
        //     imshow( myWindowName, myDestination );
        // }
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

    inline void setSourceImage( Mat& source )
    {
        mySource = source;
    }

    inline void setDestinationImage( Mat& destination )
    {
        myDestination = destination;
    }

private:
    detector_e currentDetector;
    Mat mySource;
    Mat myDestination;
    Mat detected_edges;
    Mat myImageToShow;
    String myWindowName;
    VideoCapture myCamera;
};


// static void CannyThreshold( int, void* )
// void applyCanny( Mat* src, Mat* myDestination )
void EdgeDetector::applyCanny()
{
    Mat src_gray;
    cvtColor( mySource, src_gray, COLOR_BGR2GRAY );

    //![reduce_noise]
    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size( 3, 3 ) );
    //![reduce_noise]

    //![canny]
    /// Canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold * cannyRatio, kernel_size );
    //![canny]

    /// Using Canny's output as a mask, we display our result
    //![fill]
    // myDestination = Scalar::all( 0 );
    //![fill]

    //![copyto]
    mySource.copyTo( myDestination, detected_edges );
    //![copyto]

    //![display]
    // imshow( windowName, myDestination );
    //![display]
}

// void applySobel( Mat* image, Mat* grad )
void EdgeDetector::applySobel()
{
    // Mat src;
    Mat src_gray;
    Mat gray;
    //![reduce_noise]
    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur( mySource, gray, Size( 3, 3 ), 0, 0, BORDER_DEFAULT );
    //![reduce_noise]

    //![convert_to_gray]
    // Convert the image to grayscale
    cvtColor( mySource, src_gray, COLOR_BGR2GRAY );
    //![convert_to_gray]

    //![sobel]
    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Sobel( src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT );

    /// Gradient Y
    Sobel( src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT );
    //![sobel]

    //![convert]
    // converting back to CV_8U
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );
    //![convert]

    //![blend]
    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, myDestination );
}

int main( int argc, char** argv )
{
    cv::CommandLineParser parser( argc, argv,
                                  //   "{@input   |../data/lena.jpg|input image}"
                                  "{help    h|false|show help message}" );
    bool help = parser.get<bool>( "help" );
    if( help )
    {
        cout << "The sample uses Sobel or Scharr OpenCV functions for edge detection\n\n";
        parser.printMessage();
        cout << "\nPress 'ESC' to exit program.\nPress 'R' to reset values ( ksize will be -1 equal to Scharr function )";
        return 0;
    }

    EdgeDetector edgeDetector( 0, 640, 480 );

    // if( true == parser.has( "@input" ) )
    // {
    //     String imageName = parser.get<String>( "@input" );
    //     int ret = edgeDetector.openImage( imageName );
    //     if( ret == 1 )
    //     {
    //         return 1;
    //     }
    // }
    // else
    // {
    edgeDetector.readCameraFrame();
    // }
    edgeDetector.applyTransformation();

    edgeDetector.showImage( true );

    while( true )
    {
        edgeDetector.readCameraFrame();
        edgeDetector.applyTransformation();
        edgeDetector.showImage();

        // applySobel( &src, &myDestination );
        char key = (char)waitKey( 1 );

        if( key == 27 )
        {
            break;
        }
        else if( key == 'C' || key == 'c' )
        {
            edgeDetector.setCurrentDetector( CANNY );
            // edgeDetector.applyCanny();
            // edgeDetector.showImage( false );
        }
        else if( key == 'S' || key == 's' )
        {
            edgeDetector.setCurrentDetector( SOBEL );
            // edgeDetector.applySobel();
            // edgeDetector.showImage( false );
        }
        else if( key == 'N' || key == 'n' )
        {
            edgeDetector.setCurrentDetector( NONE );
            // edgeDetector.showImage( true );
        }
    }
    return 0;
}
