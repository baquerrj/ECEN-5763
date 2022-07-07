/**
 * @file houghclines.cpp
 * @brief This program demonstrates line finding with the Hough transform
 */
#include <stdio.h>
#include <time.h>
#include <memory>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

const char* sourceImage = "Source";
const char* standardHough = "Detected Lines (in red) - Standard Hough Line Transform";
const char* probabilisticHough = "Detected Lines (in red) - Probabilistic Line Transform";
int min_threshold = 50;
int max_trackbar = 150;

int s_trackbar = max_trackbar;

int hl_threshold = min_threshold + s_trackbar;

void updateHoughThreshold( int newValue, void* object )
{
    s_trackbar = newValue;
    hl_threshold = min_threshold + s_trackbar;
}

double delta_t( struct timespec* stop, struct timespec* start )
{
    double current = ( (double)stop->tv_sec * 1000.0 ) +
        ( (double)( (double)stop->tv_nsec / 1000000.0 ) );
    double last = ( (double)start->tv_sec * 1000.0 ) +
        ( (double)( (double)start->tv_nsec / 1000000.0 ) );
    return ( current - last );
}

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
        printf("Opened camera\n\r" );

        // myKsize = ksize;
        // myScale = scale;
        // myDelta = delta;

        namedWindow( myWindowName );
        printf( "Created window: %s\n\r", myWindowName.c_str() );

        createTrackbars();
        printf( "Created trackbars: %s\n\r", myWindowName.c_str() );

    }
    ~EdgeDetector() {}

    inline void readCameraFrame()
    {
        printf("Reading camera frame\n\r");
        myCamera.read( mySource );
    }


    inline void showImage()
    {
        imshow( myWindowName, myImageToShow );
    }

    inline void showImage( Mat * src )
    {
        imshow( myWindowName, *src );
    }

    inline void createTrackbars()
    {
        createTrackbar( "HL Threshold", myWindowName, &min_threshold, max_trackbar, updateHoughThreshold, this );
        // createTrackbar( "Sobel Kernel Size:", myWindowName, &ksize, maxKsize, updateKsize, this );
        // createTrackbar( "Sobel Scale:", myWindowName, &scale, 100, updateScale, this );
        // createTrackbar( "Sobel Delta:", myWindowName, &delta, 100, updateDelta, this );
    }

    inline static void updateHoughThreshold( int newValue, void* object )
    {
        EdgeDetector* ed = (EdgeDetector*)object;

        ed->setHoughThreshold( newValue );
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

    inline void setHoughThreshold( int value )
    {
        myThreshold = value;
    }

    inline void setKsize( int value )
    {
        if( value % 2 == 0 )
        {
            myKsize = value + 1;
            setTrackbarPos( "Sobel Kernel Size:", myWindowName, myKsize );
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

    inline void printAverageFrameRates()
    {
        double deltaT = deltasCanny / framesCanny;
        deltaT = deltaT / 1000.0;
        printf( "****************** CANNY * *****************\n\r" );
        printf( "Processed %d frames in %3.4f seconds\n\r", framesCanny, deltasCanny / 1000.0 );
        printf( "Canny Average Frame Rate: %3.4f sec/frame\n\r", deltaT );
        printf( "Canny Average Frame Rate: %3.4f frames/sec (fps)\n\r", 1.0 / deltaT );
    }
    void applyHoughlines( Mat* src, Mat* dst, Mat* cdst, Mat* cdstP  );

    void applySobel();

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

    inline Mat& getSource()
    {
        return mySource;
    }

private:
    Mat mySource;
    Mat myDestination;
    Mat myImageToShow;
    String myWindowName;
    VideoCapture myCamera;
    int myThreshold;
    int myKsize;
    int myScale;
    int myDelta;

    struct timespec start;
    struct timespec stop;

    double deltasCanny;

    int framesCanny;
};

void EdgeDetector::applyHoughlines( Mat* src, Mat* dst, Mat* cdst, Mat* cdstP )
{
    printf("Applying HoughLines\n\r");
    //![edge_detection]
       // Edge detection
    Canny( *src, *dst, 50, 200, 3 );
    //![edge_detection]

    // Copy edges to the images that will display the results in BGR
    cvtColor( *dst, *cdst, COLOR_GRAY2BGR );
    *cdstP = cdst->clone();

    //![hough_lines]
    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines( *dst, lines, 1, CV_PI / 180, myThreshold, 0, 0 ); // runs the actual detection
    //![hough_lines]
    //![draw_lines]
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos( theta ), b = sin( theta );
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound( x0 + 1000 * ( -b ) );
        pt1.y = cvRound( y0 + 1000 * ( a ) );
        pt2.x = cvRound( x0 - 1000 * ( -b ) );
        pt2.y = cvRound( y0 - 1000 * ( a ) );
        line( *cdst, pt1, pt2, Scalar( 0, 0, 255 ), 3, LINE_AA );
    }
    //![draw_lines]

    //![hough_lines_p]
    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP( *dst, linesP, 1, CV_PI / 180, 50, 50, 10 ); // runs the actual detection
    //![hough_lines_p]
    //![draw_lines_p]
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        line( *cdstP, Point( l[0], l[1] ), Point( l[2], l[3] ), Scalar( 0, 0, 255 ), 3, LINE_AA );
    }
    //![draw_lines_p]

    return;
}

int main( int argc, char** argv )
{
    CommandLineParser parser( argc, argv,
                              "{camera        c|false|Use camera as source}"
                              "{video         v|false|Use video file as source}"
                              "{@videoInput    |../Dark-Room-Laser-Spot.mpeg|video source}"
                              "{help    h|false|show help message}" );
    bool help = parser.get<bool>( "help" );
    if( help )
    {
        printf( "The sample uses Sobel or Scharr OpenCV functions for edge detection\n\n" );
        parser.printMessage();
        printf("\nPress 'ESC' to exit program.\nPress 'R' to reset values ( ksize will be -1 equal to Scharr function )");
        return 0;
    }

    bool useCamera = parser.get<bool>( "camera" );
    bool useVideo = parser.get<bool>( "video" );

    String videoInput = parser.get<String>( "@videoInput" );

    // Declare the output variables
    Mat dst, cdst, cdstP;


    EdgeDetector* edgeDetector = new EdgeDetector();

    if( edgeDetector == NULL )
    {
        return -1;
    }

    // namedWindow( sourceImage );
    // namedWindow( standardHough );
    // namedWindow( probabilisticHough );

    char winInput;

    Mat src;
    // used to calculate fps in while-loop
    struct timespec start = { 0, 0 };
    struct timespec stop = { 0, 0 };
    double deltas = 0.0;

    int framesProcessed = 0;

    while( true )
    {
        edgeDetector->readCameraFrame();
        src = edgeDetector->getSource();
        if( src.empty() )
        {
            printf(" image is empty!\n\r" );
            continue;
        }
        clock_gettime( CLOCK_REALTIME, &start );
        edgeDetector->applyHoughlines( &src, &dst, &cdst, &cdstP );
        clock_gettime( CLOCK_REALTIME, &stop );
        deltas += delta_t( &stop, &start );
        framesProcessed++;

        //![imshow]
        // Show results
        edgeDetector->showImage( &src );
        // imshow( standardHough, cdst );
        // imshow( probabilisticHough, cdstP );
        //![imshow]
        if( ( winInput = waitKey( 10 ) ) == 27 )
        {
            break;
        }
    }

    double deltaTMS = deltas / framesProcessed;
    double deltaT = deltaTMS / 1000.0;
    printf( "Average Frame Rate: %3.2f ms per frame\n\r", deltaTMS );
    printf( "Average Frame Rate: %3.2f frames per sec (fps)\n\r", 1.0 / deltaT );

    if( edgeDetector != NULL )
    {
        delete edgeDetector;
    }

    return 0;
}
