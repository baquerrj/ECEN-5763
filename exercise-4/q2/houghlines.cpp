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

static const int min_threshold = 50;
static const int max_trackbar = 150;
static const int min_linelength = 50;
static const int min_maxlinegap = 50;

class LineDetector
{
    public:
    static const String SOURCE_WINDOW_NAME;
    static const String STANDARD_WINDOW_NAME;
    static const String PROBABILISTIC_WINDOW_NAME;

    static const int INITIAL_STANDARD_HOUGH_THRESHOLD = 165;
    static const int INITIAL_PROBABILISTIC_HOUGH_THRESHOLD = 20;
    static const int INITIAL_MIN_LINE_LENGTH = 10;
    static const int INITIAL_MAX_LINE_LAP = 1;

    static const int MAX_STANDARD_HOUGH_THRESHOLD = 300;
    static const int MAX_PROBABILISTIC_HOUGH_THRESHOLD = 150;
    static const int MAX_MIN_LINE_LENGTH = 100;
    static const int MAX_MAX_LINE_LAP = 25;

    public:
    LineDetector( bool useCamera = true,
                  int deviceId = 0,
                  int frameWidth = 640,
                  int frameHeight = 480,
                  const String videoInput = "",
                  const bool useTrackbars = true )
    {
        if( useCamera )
        {
            myCamera = VideoCapture( deviceId );
            if( !myCamera.isOpened() )
            {
                printf( "Error opening camera %d\n\r", deviceId );
                exit( -1 );
            }
            myCamera.set( CAP_PROP_FRAME_WIDTH, frameWidth );
            myCamera.set( CAP_PROP_FRAME_HEIGHT, frameHeight );
            printf( "Opened camera\n\r" );
        }
        else
        {
            myCamera = VideoCapture( videoInput );
        }
        myHoughLinesThreshold = INITIAL_STANDARD_HOUGH_THRESHOLD;
        myHoughLinesPThreshold = INITIAL_PROBABILISTIC_HOUGH_THRESHOLD;
        myMaxLineGap = INITIAL_MAX_LINE_LAP;
        myMinLineLength = INITIAL_MIN_LINE_LENGTH;

        namedWindow( SOURCE_WINDOW_NAME, WINDOW_NORMAL );
        resizeWindow( SOURCE_WINDOW_NAME, Size( frameWidth, frameHeight ) );
        printf( "Created window: %s\n\r", SOURCE_WINDOW_NAME.c_str() );

        namedWindow( STANDARD_WINDOW_NAME, WINDOW_NORMAL );
        resizeWindow( STANDARD_WINDOW_NAME, Size( frameWidth, frameHeight ) );
        printf( "Created window: %s\n\r", STANDARD_WINDOW_NAME.c_str() );

        namedWindow( PROBABILISTIC_WINDOW_NAME, WINDOW_NORMAL );
        resizeWindow( PROBABILISTIC_WINDOW_NAME, Size( frameWidth, frameHeight ) );
        printf( "Created window: %s\n\r", PROBABILISTIC_WINDOW_NAME.c_str() );

        if( useTrackbars )
        {
            createTrackbars();
            printf( "Created trackbars for (%s) and (%s)\n\r", STANDARD_WINDOW_NAME.c_str(), PROBABILISTIC_WINDOW_NAME.c_str() );
        }

    }
    ~LineDetector()
    {
        destroyAllWindows();
    }

    inline void readCameraFrame()
    {
        myCamera.read( mySource );
    }

    inline bool isFrameEmpty()
    {
        return mySource.empty();
    }


    inline void showSourceImage()
    {
        imshow( SOURCE_WINDOW_NAME, mySource );
    }

    inline void showStandardTransform()
    {
        imshow( STANDARD_WINDOW_NAME, myStandardHoughTransform );
    }

    inline void showProbabilisticTransform()
    {
        imshow( PROBABILISTIC_WINDOW_NAME, myProbabilisticHoughTransform );
    }

    inline void createTrackbars()
    {
        createTrackbar( "HL Threshold", STANDARD_WINDOW_NAME,
                        &myHoughLinesThreshold,
                        MAX_STANDARD_HOUGH_THRESHOLD,
                        updateHoughThreshold,
                        this );
        createTrackbar( "HL P Threshold:",
                        PROBABILISTIC_WINDOW_NAME,
                        &myHoughLinesPThreshold,
                        MAX_PROBABILISTIC_HOUGH_THRESHOLD,
                        updateHoughLinesPThreshold,
                        this );
        createTrackbar( "HL P Min Line Length:",
                        PROBABILISTIC_WINDOW_NAME,
                        &myMinLineLength,
                        MAX_MIN_LINE_LENGTH,
                        updateMinLineLength,
                        this );
        createTrackbar( "HL P Max Line Gap:",
                        PROBABILISTIC_WINDOW_NAME,
                        &myMaxLineGap,
                        MAX_MAX_LINE_LAP,
                        updateMaxLineGap,
                        this );
    }

    inline static void updateHoughThreshold( int newValue, void* object )
    {
        LineDetector* ed = (LineDetector*)object;

        ed->setHoughLinesThreshold( newValue );
    }

    inline static void updateHoughLinesPThreshold( int newValue, void* object )
    {
        LineDetector* ed = (LineDetector*)object;

        ed->setHoughLinesPThreshold( newValue );
    }

    inline static void updateMinLineLength( int newValue, void* object )
    {
        LineDetector* ed = (LineDetector*)object;

        ed->setMinLineLength( newValue );
    }

    inline static void updateMaxLineGap( int newValue, void* object )
    {
        LineDetector* ed = (LineDetector*)object;

        ed->setMaxLineGap( newValue );
    }

    inline void setHoughLinesThreshold( int value )
    {
        myHoughLinesThreshold = value;
    }

    inline void setHoughLinesPThreshold( int value )
    {
        myHoughLinesPThreshold = value;
    }
    inline void setMinLineLength( int value )
    {
        myMinLineLength = value;
    }
    inline void setMaxLineGap( int value )
    {
        myMaxLineGap = value;
    }

    void applyHoughlines();

    private:
    Mat mySource;
    VideoCapture myCamera;
    int myHoughLinesThreshold;
    int myHoughLinesPThreshold;
    int myMinLineLength;
    int myMaxLineGap;

    Mat tmp;
    Mat myStandardHoughTransform;
    Mat myProbabilisticHoughTransform;
};

const String LineDetector::SOURCE_WINDOW_NAME = "Source";
const String LineDetector::STANDARD_WINDOW_NAME = "Detected Lines (in red) - Standard Hough Line Transform";
const String LineDetector::PROBABILISTIC_WINDOW_NAME = "Detected Lines (in red) - Probabilistic Line Transform";

void LineDetector::applyHoughlines()
{
    // Edge detection
    Canny( mySource, tmp, 50, 200, 3 );
    //![edge_detection]

    // Copy edges to the images that will display the results in BGR
    cvtColor( tmp, myStandardHoughTransform, COLOR_GRAY2BGR );
    myProbabilisticHoughTransform = myStandardHoughTransform.clone();

    //![hough_lines]
    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines( tmp, lines, 1, CV_PI / 180, myHoughLinesThreshold, 0, 0 ); // runs the actual detection
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
        line( myStandardHoughTransform, pt1, pt2, Scalar( 0, 0, 255 ), 3, LINE_AA );
    }
    //![draw_lines]

    //![hough_lines_p]
    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    printf( "HLP with %d, %d, %d\n\r", myHoughLinesPThreshold, myMinLineLength, myMaxLineGap );
    HoughLinesP( tmp, linesP, 1, CV_PI / 180, myHoughLinesPThreshold, myMinLineLength, myMaxLineGap ); // runs the actual detection
    //![hough_lines_p]
    //![draw_lines_p]
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        line( myProbabilisticHoughTransform, Point( l[0], l[1] ), Point( l[2], l[3] ), Scalar( 0, 0, 255 ), 3, LINE_AA );
    }
    //![draw_lines_p]

    return;
}

int main( int argc, char** argv )
{
    CommandLineParser parser( argc, argv,
                              "{camera          c|false|Use camera as source. If omitted, path to file must be supplied.}"
                              "{video           v|./NIR-front-facing/GP010639.MP4|video source}"
                              "{useTrackbars    t|false|Use trackbars}"
                              "{help            h|false|show help message}" );
    bool help = parser.get<bool>( "help" );
    if( help )
    {
        printf( "The program uses the standard and probabilistic Hough algorithms to detect lines\n" );
        parser.printMessage();
        printf( "Press the ESC key to exit the program.\n" );
        return 0;
    }

    bool useCamera = parser.get<bool>( "camera" );
    bool useTrackbars = parser.get<bool>( "useTrackbars" );

    String videoInput = parser.get<String>( "video" );

    LineDetector* detector = new LineDetector( useCamera, 0, 640, 480, videoInput, useTrackbars );

    if( detector == NULL )
    {
        return -1;
    }

    char winInput;

    int framesProcessed = 0;

    while( true )
    {
        detector->readCameraFrame();
        if( detector->isFrameEmpty() )
        {
            break;
        }
        detector->applyHoughlines();
        framesProcessed++;

        //![imshow]
        // Show results
        detector->showStandardTransform();
        detector->showProbabilisticTransform();
        detector->showSourceImage();
        //![imshow]
        if( ( winInput = waitKey( 2 ) ) == 27 )
        {
            break;
        }
    }

    if( detector != NULL )
    {
        delete detector;
    }

    return 0;
}
