/*!
 * @file lanedetector.h
 * @author Roberto J Baquerizo (roba8460@colorado.edu)
 * @brief
 * @version 1.0
 * @date 2022-08-04
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __LANE_DETECTOR_H__
#define __LANE_DETECTOR_H__

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
    /*!
     * @brief Struct holding all information needed to annotate an image
     *
     */
    struct frame_s
    {
        cv::Mat currentRawImage;        //!< Raw image to process/annotate
        cv::Mat currentAnnotatedImage;  //!< Annotated image to display
        cv::Point leftPt1;              //!< Point 1 defining the left lane
        cv::Point leftPt2;              //!< Point 2 defining the left lane
        cv::Point rightPt1;             //!< Point 1 defining the right lane
        cv::Point rightPt2;             //!< Point 2 defining the right lane
        std::vector< cv::Rect > vehicle;    //!< Vector of detected vehicles
        uint64_t number;                //!< Frame numbers, 0th frame, 1st frame, etc.
    };

    public:
    static const cv::String SOURCE_WINDOW_NAME;
    static const cv::String DETECTED_LANES_IMAGE;
    static const cv::String DETECTED_VEHICLES_IMAGE;
    static const cv::String CAR_CLASSIFIER;

    static const int RING_BUFFER_SIZE = 150;    //!< Defines buffer size for ring buffers

    static const int DEFAULT_FRAME_WIDTH = 1280;    //!< Default frame width
    static const int DEFAULT_FRAME_HEIGHT = 720;    //!< Default frame height
    static const int DEFAULT_DEVICE_ID = 0;         //!< Default device ID, if using camera as input
    static const bool DEFAULT_USE_CAMERA = false;   //!< Default to not use camera

    static const int LEFT_MINIMUM_THETA = 10;       //!< Minimum theta for left lane
    static const int LEFT_MAXIMUM_THETA = 65;       //!< Maximum theta for left lane
    static const int RIGHT_MINIMUM_THETA = 115;     //!< Minimum theta for right lane
    static const int RIGHT_MAXIMUM_THETA = 170;     //!< Maximum theta for right lane

    static const int EMPTY_FRAMES_PERSISTENCY_CHECK = 5;    //!< Number of empty frames before quitting

    public:
    /*!
     * @brief Construct a new Line Detector object
     *
     * @param configData thread configuration information
     * @param deviceId
     * @param videoFilename
     * @param saveFrames
     * @param outputDirectory
     * @param showWindows
     * @param frameWidth
     * @param frameHeight
     */
    LineDetector( const ThreadConfigData* configData,
                  int deviceId,
                  const cv::String videoFilename = "",
                  bool saveFrames = false,
                  const cv::String outputDirectory = "",
                  bool showWindows = false,
                  int frameWidth = DEFAULT_FRAME_WIDTH,
                  int frameHeight = DEFAULT_FRAME_HEIGHT );

    /*!
     * @brief Destroy the Line Detector object
     *
     */
    virtual ~LineDetector();

    /*!
     * @brief Load Haar Cascacde XML classifier
     *
     * @param classifier
     * @return true
     * @return false
     */
    bool loadClassifier( const cv::String& classifier );

    /*!
     * @brief Create windows to display images, if necessary
     *
     */
    void createWindows();

    /*!
     * @brief Print average frame rate of each processing thread
     *
     */
    void printFrameRates();

    /*!
     * @brief Read frame from the video file
     *
     */
    void readFrame();

    /*!
     * @brief Update window with annotated image
     *
     */
    void updateDisplayWindow();

    /*!
     * @brief Get the frame rate from VideoCapture
     *
     * @return int
     */
    int getFrameRate();

    /*!
     * @brief Get the width for the
     *
     * @return int
     */
    int getFrameWidth();
    int getFrameHeight();
    void prepareImage();
    void detectLanes();
    bool intersection( cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f& r );
    void findLeftLane( cv::Vec4i left, frame_s& f );
    void findRightLane( cv::Vec4i right, frame_s& f );
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

    bool createdOk()
    {
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
    int myFrameWidth;
    int myFrameHeight;
    int myDeviceId;
    bool showWindows;
    bool saveFrames;
    double carsDeltaTimes;
    double lanesDeltaTimes;
    double annotationDeltaTimes;

    uint64_t carsDetected;
    uint64_t leftLanesDetected;
    uint64_t rightLanesDetected;

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

#endif // __LANE_DETECTOR_H__