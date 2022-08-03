#ifndef __LANE_DETECTOR_HPP__
#define __LANE_DETECTOR_HPP__

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"


#include <semaphore.h>
#include "RingBuffer.h"

// Forward declarartions
class CyclicThread;
struct ThreadConfigData;

class LineDetector
{
    struct frame_s
    {
        cv::Mat currentRawImage;   // frame that goes with this frame
        cv::Mat currentAnnotatedImage;   // frame that goes with this frame
        cv::Point leftPt1;
        cv::Point leftPt2;
        cv::Point rightPt1;
        cv::Point rightPt2;
        std::vector< cv::Rect > vehicle;
        uint64_t number;
    };

    public:
    static const cv::String SOURCE_WINDOW_NAME;
    static const cv::String DETECTED_LANES_IMAGE;
    static const cv::String DETECTED_VEHICLES_IMAGE;
    static const cv::String CAR_CLASSIFIER;

    static const int RING_BUFFER_SIZE = 100;

    static const int DEFAULT_FRAME_WIDTH = 1280;
    static const int DEFAULT_FRAME_HEIGHT = 720;
    static const int DEFAULT_DEVICE_ID = 0;
    static const bool DEFAULT_USE_CAMERA = false;

    static const int LEFT_MINIMUM_THETA = 10;
    static const int LEFT_MAXIMUM_THETA = 110;
    static const int RIGHT_MINIMUM_THETA = 115;
    static const int RIGHT_MAXIMUM_THETA = 170;

    static const int EMPTY_FRAMES_PERSISTENCY_CHECK = 5;

    public:
    LineDetector( const ThreadConfigData* configData,
                  int deviceId,
                  const cv::String videoFilename = "",
                  bool saveFrames = false,
                  const cv::String outputDirectory = "",
                  bool showWindows = false,
                  int frameWidth = DEFAULT_FRAME_WIDTH,
                  int frameHeight = DEFAULT_FRAME_HEIGHT );

    virtual ~LineDetector();

    bool loadClassifier( const cv::String& classifier );

    void createWindows();
    void printFrameRates();
    void readFrame();

    bool isFrameEmpty();

    void showSourceImage();

    void showLanesImage();

    void showVehiclesImage();

    cv::Mat getVehiclesImage();

    int getFrameRate();

    int getFrameWidth();

    int getFrameHeight();

    void prepareImage();

    void detectLanes();

    bool intersection( cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f& r );

    void findLeftLane( cv::Vec4i left, frame_s &f );
    void findRightLane( cv::Vec4i right, frame_s& f );
    bool isInsideRoi( cv::Point p );
    void detectCars();

    void writeFrameToVideo();

    void annotateImage();

    virtual void shutdown();
    virtual bool isAlive();
    virtual bool isCaptureThreadAlive();
    virtual bool isLineThreadAlive();
    virtual bool isCarThreadAlive();
    virtual bool isAnnotationThreadAlive();

    virtual pthread_t getCaptureThreadId();

    bool createdOk() {
        return myCreatedOk;
    }

    static void* executeCapture( void* context );
    static void* executeLine( void* context );
    static void* executeCar( void* context );
    static void* executeAnnotation( void* context );


    private:
    bool foundLeft;
    bool foundRight;
    bool myCreatedOk;
    bool lanesReady;
    bool carsReady;
    int myFrameWidth;
    int myFrameHeight;
    int myDeviceId;
    bool showWindows;
    bool saveFrames;
    double carsDeltaTimes;
    double lanesDeltaTimes;
    double annotationDeltaTimes;

    uint64_t carsThreadFrames;
    uint64_t lanesThreadFrames;
    uint64_t annotationThreadFrames;

    cv::String myVideoFilename;
    cv::String myOutputDirectory;

    struct timespec carsStart;
    struct timespec carsStop;

    struct timespec lanesStart;
    struct timespec lanesStop;

    struct timespec annotationStart;
    struct timespec annotationStop;

    uint16_t numberOfEmptyFrames;

    cv::VideoCapture myVideoCapture;
    // cv::VideoWriter myVideoWriter;


    cv::Mat rawImage;
    cv::Mat imageToProcess;

    cv::Mat tmp;

    cv::CascadeClassifier myClassifier;

    RingBuffer< cv::Mat >* p_myRawBuffer;
    RingBuffer< cv::Mat >* p_bufferForCarDetection;
    RingBuffer< cv::Mat >* p_myReadyToAnnotateBuffer;
    RingBuffer< frame_s >* p_myFinalBuffer;

    cv::Point roiPoints[ 4 ];
    cv::Point carPoints[ 4 ];
    cv::Mat roi;

    RingBuffer< frame_s >* frames;

    pthread_mutex_t frameLock;

    pthread_mutex_t lock;
    pthread_mutex_t roiLock;
    pthread_mutex_t carRingLock;

    protected:
    // Capture Thread
    std::string captureThreadName;
    bool captureThreadAlive;
    CyclicThread* captureThread;

    // Line Detection Thread
    std::string lineDetectionThreadName;
    bool lineDetectionThreadAlive;
    CyclicThread* lineDetectionThread;

    // Car Detection Thread
    std::string carDetectionThreadName;
    bool carDetectionThreadAlive;
    CyclicThread* carDetectionThread;

    // Annotation Thread
    std::string annotationThreadName;
    bool annotationThreadAlive;
    CyclicThread* annotationThread;
};


inline bool LineDetector::isAlive()
{
    return ( isCaptureThreadAlive() ||
             isLineThreadAlive() ||
             isCarThreadAlive() ||
             isAnnotationThreadAlive() );
}

inline int LineDetector::getFrameRate()
{
    return myVideoCapture.get( cv::CAP_PROP_FPS );
}

inline int LineDetector::getFrameWidth()
{
    return myVideoCapture.get( cv::CAP_PROP_FRAME_WIDTH );
}

inline int LineDetector::getFrameHeight()
{
    return myVideoCapture.get( cv::CAP_PROP_FRAME_HEIGHT );
}

#endif // __LANE_DETECTOR_HPP__