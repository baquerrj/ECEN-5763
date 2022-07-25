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
                  const String videoFilename = "",
                  bool writeOutputVideo = false,
                  const String outputVideoFilename = "",
                  int frameWidth = DEFAULT_FRAME_WIDTH,
                  int frameHeight = DEFAULT_FRAME_HEIGHT )
    {
        myFrameHeight = frameHeight;
        myFrameWidth = frameWidth;
        myDeviceId = deviceId;
        myVideoFilename = videoFilename;

        myHoughLinesPThreshold = INITIAL_PROBABILISTIC_HOUGH_THRESHOLD;
        myMaxLineGap = INITIAL_MAX_LINE_LAP;
        myMinLineLength = INITIAL_MIN_LINE_LENGTH;

        myVideoCapture = VideoCapture( myVideoFilename );

        myVideoCapture.set( CAP_PROP_FRAME_HEIGHT, (double)frameHeight );
        myVideoCapture.set( CAP_PROP_FRAME_WIDTH, (double)frameWidth );

        createWindows();

        if( writeOutputVideo )
        {
            myOutputVideoFilename = outputVideoFilename;
            if( myOutputVideoFilename.empty() or myOutputVideoFilename == "" )
            {
                myOutputVideoFilename = "output.avi";
            }
            myVideoWriter.open( myOutputVideoFilename,
                                VideoWriter::fourcc( 'M', 'J', 'P', 'G' ),
                                getFrameRate(),
                                Size( getFrameWidth(), getFrameHeight() ),
                                true );
        }
    }

    ~LineDetector()
    {
        if( myVideoCapture.isOpened() )
        {
            myVideoCapture.release();
        }

        if( myVideoWriter.isOpened() )
        {
            myVideoWriter.release();
        }

        destroyAllWindows();
    }

    bool loadClassifier( const String& classifier );

    void createWindows();

    void readFrame();

    bool isFrameEmpty();

    void showSourceImage();

    void showLanesImage();

    void showVehiclesImage();

    void setHoughLinesPThreshold( int value );

    void setMinLineLength( int value );

    void setMaxLineGap( int value );

    Mat getVehiclesImage();

    int getFrameRate();

    int getFrameWidth();

    int getFrameHeight();

    void prepareImage();

    void detectLanes();

    void detectCars();

    void writeFrameToVideo();

    private:
    Mat mySource;
    VideoCapture myVideoCapture;
    VideoWriter myVideoWriter;
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
    String myOutputVideoFilename;

    CascadeClassifier myClassifier;
};

#endif // __LANE_DETECTOR_HPP__