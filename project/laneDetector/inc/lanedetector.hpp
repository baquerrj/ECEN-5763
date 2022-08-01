#ifndef __LANE_DETECTOR_HPP__
#define __LANE_DETECTOR_HPP__

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"


#include <semaphore.h>
#include "RingBuffer.h"

using namespace cv;

// Forward declarartions
class CyclicThread;
struct ThreadConfigData;


class LineDetector
{
    public:
    static const String SOURCE_WINDOW_NAME;
    static const String DETECTED_LANES_IMAGE;
    static const String DETECTED_VEHICLES_IMAGE;
    static const String CAR_CLASSIFIER;

    static const int RING_BUFFER_SIZE = 100;
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
    LineDetector( const ThreadConfigData* configData,
                  int deviceId,
                  const String videoFilename = "",
                  bool writeOutputVideo = false,
                  const String outputVideoFilename = "",
                  int frameWidth = DEFAULT_FRAME_WIDTH,
                  int frameHeight = DEFAULT_FRAME_HEIGHT );

    virtual ~LineDetector();

    bool loadClassifier( const String& classifier );

    void createWindows();

    void readFrame();

    bool isFrameEmpty();

    void showSourceImage();

    void showLanesImage();

    void showVehiclesImage();

    Mat getVehiclesImage();

    int getFrameRate();

    int getFrameWidth();

    int getFrameHeight();

    void prepareImage();

    void detectLanes();

    bool getIntersection( Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f& r );

    void findLeftLane( Vec4i left );
    void findRightLane( Vec4i right );
    bool isInsideRoi( Point p );
    void detectCars();

    void writeFrameToVideo();

    bool newFrameReady();

    virtual void shutdown();

    virtual bool isAlive();

    virtual bool isCaptureThreadAlive();
    virtual bool isLineThreadAlive();
    virtual bool isCarThreadAlive();

    virtual pthread_t getCaptureThreadId();

    virtual sem_t* getCaptureSemaphore();

    bool createdOk() {
        return myCreatedOk;
    }

    inline uint64_t getFramesProcessed()
    {
        return framesProcessed;
    }

    static void* executeCapture( void* context );
    static void* executeLine( void* context );
    static void* executeCar( void* context );


    private:
    Mat annot;
    VideoCapture myVideoCapture;
    VideoWriter myVideoWriter;
    int myHoughLinesPThreshold;
    int myMinLineLength;
    int myMaxLineGap;
    bool myNewFrameReady;
    bool myCreatedOk;

    Mat rawImage;
    Mat imageToProcess;

    Mat tmp;
    Mat mySourceCopy;
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

    RingBuffer< Mat >* p_myRawBuffer;
    RingBuffer< Mat >* p_myGrayscaleBuffer;
    RingBuffer< Mat >* p_myFinalBuffer;

    uint64_t framesProcessed;

    Point roiPoints[ 4 ];
    Mat roi;
    Point leftPt1;
    Point leftPt2;
    Point rightPt1;
    Point rightPt2;


    bool foundLeft;
    bool foundRight;

    protected:
    // Capture Thread
    // ThreadConfigData captureConfig;
    std::string captureThreadName;
    bool captureThreadAlive;
    sem_t captureThreadSem;
    CyclicThread* captureThread;

    // Line Detection Thread
    // ThreadConfigData lineConfig;
    std::string lineDetectionThreadName;
    bool lineDetectionThreadAlive;
    sem_t lineDetectionThreadSem;
    CyclicThread* lineDetectionThread;

    // Car Detection Thread
    // ThreadConfigData carCondig;
    std::string carDetectionThreadName;
    bool carDetectionThreadAlive;
    sem_t carDetectionThreadSem;
    CyclicThread* carDetectionThread;
};


inline bool LineDetector::isAlive()
{
    return ( isCaptureThreadAlive() || isLineThreadAlive() || isCarThreadAlive() );
}

inline sem_t* LineDetector::getCaptureSemaphore()
{
    return &captureThreadSem;
}

inline Mat LineDetector::getVehiclesImage()
{
    return myVehiclesImage;
}

inline int LineDetector::getFrameRate()
{
    return myVideoCapture.get( CAP_PROP_FPS );
}

inline int LineDetector::getFrameWidth()
{
    return myVideoCapture.get( CAP_PROP_FRAME_WIDTH );
}

inline int LineDetector::getFrameHeight()
{
    return myVideoCapture.get( CAP_PROP_FRAME_HEIGHT );
}

inline void LineDetector::showVehiclesImage()
{
    imshow( DETECTED_VEHICLES_IMAGE, myVehiclesImage );
}

inline bool LineDetector::newFrameReady()
{
    return myNewFrameReady;
}

// inline bool LineDetector::isFrameEmpty()
// {
//     return mySource.empty();
// }

// inline void LineDetector::showSourceImage()
// {
//     if( not mySource.empty() )
//     {
//         imshow( SOURCE_WINDOW_NAME, mySource );
//     }
// }

// inline void LineDetector::showLanesImage()
// {
//     if( not p_myFinalBuffer->isEmpty() )
//     {
//         imshow( DETECTED_LANES_IMAGE, p_myFinalBuffer->dequeue() );
//     }
// }

#endif // __LANE_DETECTOR_HPP__