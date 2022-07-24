#ifndef __LANE_DETECTOR_HPP__
#define __LANE_DETECTOR_HPP__

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

static const int min_threshold = 50;
static const int min_linelength = 50;
static const int min_maxlinegap = 50;

using namespace cv;


class LineDetector
{
    public:
    static const String SOURCE_WINDOW_NAME;
    static const String DETECTED_LANES_IMAGE;
    static const String DETECTED_VEHICLES_IMAGE;
    static const String CAR_CLASSIFIER;

    static const int INITIAL_PROBABILISTIC_HOUGH_THRESHOLD = 20;
    static const int INITIAL_MIN_LINE_LENGTH = 10;
    static const int INITIAL_MAX_LINE_LAP = 1;

    static const int MAX_STANDARD_HOUGH_THRESHOLD = 300;
    static const int MAX_PROBABILISTIC_HOUGH_THRESHOLD = 150;
    static const int MAX_MIN_LINE_LENGTH = 100;
    static const int MAX_MAX_LINE_LAP = 25;

    static const int DEFAULT_FRAME_WIDTH = 1280;
    static const int DEFAULT_FRAME_HEIGHT = 720;
    static const int DEFAULT_DEVICE_ID = 0;
    static const bool DEFAULT_USE_CAMERA = false;

    public:
    LineDetector( int deviceId,
                  int frameWidth,
                  int frameHeight,
                  const String videoFilename = "" )
    {
        myFrameHeight = frameHeight;
        myFrameWidth = frameWidth;
        myDeviceId = deviceId;
        myVideoFilename = videoFilename;

        myHoughLinesPThreshold = INITIAL_PROBABILISTIC_HOUGH_THRESHOLD;
        myMaxLineGap = INITIAL_MAX_LINE_LAP;
        myMinLineLength = INITIAL_MIN_LINE_LENGTH;

        initialize();
    }

    LineDetector( int deviceId,
                  const String videoFilename = "" )
    {
        myFrameHeight = DEFAULT_FRAME_HEIGHT;
        myFrameWidth = DEFAULT_FRAME_WIDTH;
        myDeviceId = deviceId;
        myVideoFilename = videoFilename;

        myHoughLinesPThreshold = INITIAL_PROBABILISTIC_HOUGH_THRESHOLD;
        myMaxLineGap = INITIAL_MAX_LINE_LAP;
        myMinLineLength = INITIAL_MIN_LINE_LENGTH;

        initialize();
    }

    ~LineDetector()
    {
        if( myVideoCapture.isOpened() )
        {
            myVideoCapture.release();
        }
        destroyAllWindows();
    }

    inline void initialize()
    {
        myVideoCapture = VideoCapture( myVideoFilename );

        createWindows( myFrameWidth, myFrameHeight );
    }

    inline bool loadClassifier( const String classifier )
    {
        myClassifier.load( classifier );

        if( myClassifier.empty() )
        {
            printf( "Failed to load classifier from %s\n\r", classifier.c_str() );
            return false;
        }
        else
        {
            return true;
        }
    }

    inline void createWindows( int frameWidth, int frameHeight )
    {

        namedWindow( SOURCE_WINDOW_NAME, WINDOW_NORMAL );
        resizeWindow( SOURCE_WINDOW_NAME, Size( frameWidth, frameHeight ) );
        printf( "Created window: %s\n\r", SOURCE_WINDOW_NAME.c_str() );

        namedWindow( DETECTED_LANES_IMAGE, WINDOW_NORMAL );
        resizeWindow( DETECTED_LANES_IMAGE, Size( frameWidth, frameHeight ) );
        printf( "Created window: %s\n\r", DETECTED_LANES_IMAGE.c_str() );

        namedWindow( DETECTED_VEHICLES_IMAGE, WINDOW_NORMAL );
        resizeWindow( DETECTED_VEHICLES_IMAGE, Size( frameWidth, frameHeight ) );
        printf( "Created window: %s\n\r", DETECTED_VEHICLES_IMAGE.c_str() );
    }

    inline void readFrame()
    {
        myVideoCapture.read( mySource );
    }

    inline bool isFrameEmpty()
    {
        return mySource.empty();
    }


    inline void showSourceImage()
    {
        imshow( SOURCE_WINDOW_NAME, mySource );
    }

    inline void showLanesImage()
    {
        imshow( DETECTED_LANES_IMAGE, myLanesImage );
    }

    inline void showVehiclesImage()
    {
        imshow( DETECTED_VEHICLES_IMAGE, myVehiclesImage );
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

    void prepareImage();

    void detectLanes();

    void detectCars();

    private:
    Mat mySource;
    VideoCapture myVideoCapture;
    int myHoughLinesPThreshold;
    int myMinLineLength;
    int myMaxLineGap;

    Mat tmp;
    Mat myLanesImage;
    Mat myVehiclesImage;
    Mat myCannyOutput;
    Mat myGrayscaleImage;

    int myFrameWidth;
    int myFrameHeight;
    int myDeviceId;
    String myVideoFilename;

    public:
    CascadeClassifier myClassifier;
};

#endif // __LANE_DETECTOR_HPP__