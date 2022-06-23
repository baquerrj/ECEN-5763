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

Mat dst, detected_edges;

// static void CannyThreshold( int, void* )
void applyCanny( Mat * src, Mat * dst )
{
    Mat src_gray;
    cvtColor( *src, src_gray, COLOR_BGR2GRAY );

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
    // dst = Scalar::all( 0 );
    //![fill]

    //![copyto]
    src->copyTo( *dst, detected_edges );
    //![copyto]

    //![display]
    // imshow( windowName, dst );
    //![display]
}

void applySobel( Mat* image, Mat* grad )
{
    Mat src;
    Mat src_gray;
    //![reduce_noise]
    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur( *image, src, Size( 3, 3 ), 0, 0, BORDER_DEFAULT );
    //![reduce_noise]

    //![convert_to_gray]
    // Convert the image to grayscale
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
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
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, *grad );
}

int main( int argc, char** argv )
{
    cv::CommandLineParser parser( argc, argv,
                                  "{@input   |../data/lena.jpg|input image}"
                                  "{help    h|false|show help message}" );
    bool help = parser.get<bool>( "help" );
    if( help )
    {
        cout << "The sample uses Sobel or Scharr OpenCV functions for edge detection\n\n";
        parser.printMessage();
        cout << "\nPress 'ESC' to exit program.\nPress 'R' to reset values ( ksize will be -1 equal to Scharr function )";
        return 0;
    }

    String imageName = parser.get<String>( "@input" );

    Mat src;
    Mat dst;
    src = imread( imageName, IMREAD_COLOR );
    if( src.empty() )
    {
        printf( "Error opening image: %s\n", imageName.c_str() );
        return 1;
    }
    imshow( windowName, src );

    while( true )
    {
        // applySobel( &src, &dst );
        char key = (char)waitKey( 0 );

        if( key == 27 )
        {
            break;
        }
        else if( key == 'C' || key == 'c' )
        {
            applyCanny( &src, &dst );
            imshow( windowName, dst );
        }
        else if( key == 'S' || key == 's' )
        {
            applySobel( &src, &dst );
            imshow( windowName, dst );
        }
        else if( key == 'N' || key == 'n' )
        {
            imshow( windowName, src );
        }
    }
    return 0;
}
