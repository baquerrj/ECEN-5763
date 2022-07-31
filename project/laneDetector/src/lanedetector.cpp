/**
 * @file houghclines.cpp
 * @brief This program demonstrates line finding with the Hough transform
 */

#include "lanedetector.hpp"
#include "configuration.hpp"
#include "thread.hpp"
#include "thread_utils.hpp"
#include "string.h"
using namespace std;

#include "Logger.h"
#include "RingBuffer.h"
extern pthread_mutex_t ringLock;
extern pthread_mutex_t imageLock;

const String LineDetector::SOURCE_WINDOW_NAME = "Source";
const String LineDetector::DETECTED_LANES_IMAGE = "Detected Lanes (in red)";
const String LineDetector::DETECTED_VEHICLES_IMAGE = "Detected Vehicles";
const String LineDetector::CAR_CLASSIFIER = "cars.xml";

LineDetector::LineDetector( const ThreadConfigData* configData,
                            int deviceId,
                            const String videoFilename,
                            bool writeOutputVideo,
                            const String outputVideoFilename,
                            int frameWidth,
                            int frameHeight )
{
    myCreatedOk = true;
    myFrameHeight = frameHeight;
    myFrameWidth = frameWidth;
    myDeviceId = deviceId;
    myVideoFilename = videoFilename;

    myHoughLinesPThreshold = INITIAL_PROBABILISTIC_HOUGH_THRESHOLD;
    myMaxLineGap = INITIAL_MAX_LINE_LAP;
    myMinLineLength = INITIAL_MIN_LINE_LENGTH;

    myVideoCapture = VideoCapture( myVideoFilename );

    myVideoCapture.set( CAP_PROP_FRAME_HEIGHT, ( double )frameHeight );
    myVideoCapture.set( CAP_PROP_FRAME_WIDTH, ( double )frameWidth );

    createWindows();

    p_myRawBuffer = new RingBuffer< Mat >( 25 );

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


    if( not loadClassifier( LineDetector::CAR_CLASSIFIER ) )
    {
        LogFatal( "Unable to load Haar classifier!" );
        myCreatedOk = false;
    }

    if( myCreatedOk )
    {
        myNewFrameReady = false;
        ThreadConfigData captureConfig = configData[ 0 ];
        captureThread = new CyclicThread( captureConfig,
                                          LineDetector::executeCapture,
                                          this,
                                          true );
        if( NULL == captureThread )
        {
            LogError( "Could not allocated memory for %s", captureConfig.threadName.c_str() );
            myCreatedOk = false;
        }
        if( not captureThread->isThreadAlive() )
        {
            LogError( "Could not start thread for %s", captureConfig.threadName.c_str() );
            myCreatedOk = false;
        }

        if( myCreatedOk )
        {
            ThreadConfigData lineConfig = configData[ 1 ];
            lineDetectionThread = new CyclicThread( lineConfig,
                                                    LineDetector::executeLine,
                                                    this,
                                                    true );
            if( NULL == lineDetectionThread )
            {
                LogError( "Could not allocated memory for %s", lineConfig.threadName.c_str() );
                myCreatedOk = false;
            }
            if( not lineDetectionThread->isThreadAlive() )
            {
                LogError( "Could not start thread for %s", lineConfig.threadName.c_str() );
                myCreatedOk = false;
            }
            if( myCreatedOk )
            {
                ThreadConfigData carConfig = configData[ 2 ];
                carDetectionThread = new CyclicThread( carConfig,
                                                        LineDetector::executeCar,
                                                        this,
                                                        true );
                if( NULL == carDetectionThread )
                {
                    LogError( "Could not allocated memory for %s", carConfig.threadName.c_str() );
                    myCreatedOk = false;
                }
                if( not carDetectionThread->isThreadAlive() )
                {
                    LogError( "Could not start thread for %s", carConfig.threadName.c_str() );
                    myCreatedOk = false;
                }
            }
        }
    }
}

LineDetector::~LineDetector()
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

    if( captureThread )
    {
        delete captureThread;
        captureThread = NULL;
    }

    if( lineDetectionThread )
    {
        delete lineDetectionThread;
        lineDetectionThread = NULL;
    }

    if( carDetectionThread )
    {
        delete carDetectionThread;
        carDetectionThread = NULL;
    }

    if( p_myRawBuffer )
    {
        delete p_myRawBuffer;
        p_myRawBuffer = NULL;
    }
}

void* LineDetector::executeCapture( void* context )
{
    ( ( LineDetector* )context )->readFrame();
    return NULL;
}

void LineDetector::readFrame()
{
    if( abortS1 )
    {
        LogDebug( "Aborting %s.", captureThread->getName() );
        captureThread->shutdown();
        return;
    }
    myNewFrameReady = false;
    pthread_mutex_lock( &ringLock );
    if( p_myRawBuffer->isFull() )
    {
        myNewFrameReady = false;
        pthread_mutex_unlock( &ringLock );
    }
    else
    {
        myVideoCapture.read( mySource );
        p_myRawBuffer->enqueue( mySource );
        pthread_mutex_unlock( &ringLock );
        myNewFrameReady = true;
        LogDebug( "New image ready!" );
    }
}

void* LineDetector::executeLine( void* context )
{
    ( ( LineDetector* )context )->detectLanes();
    return NULL;
}

void* LineDetector::executeCar( void* context )
{
    ( ( LineDetector* )context )->detectCars();
    return NULL;
}

void LineDetector::prepareImage()
{
    if( not p_myRawBuffer->isEmpty() )
    {
        pthread_mutex_lock( &ringLock );
        mySourceCopy = p_myRawBuffer->dequeue();
        pthread_mutex_unlock( &ringLock );
        myLanesImage = mySourceCopy.clone();
        cvtColor( mySourceCopy, myGrayscaleImage, COLOR_RGB2GRAY );
        GaussianBlur( mySourceCopy, tmp, Size( 5, 5 ), 0, 0, BORDER_DEFAULT );
        Canny( tmp, myCannyOutput, 40, 120, 3, true );
        // cvtColor( myCannyOutput, myLanesImage, COLOR_GRAY2RGB );
        // cvtColor( myCannyOutput, myVehiclesImage, COLOR_GRAY2RGB );
    }
    else
    {
        return;
    }
}

void LineDetector::detectLanes()
{
    if( abortS2 )
    {
        LogDebug( "Aborting %s.", lineDetectionThread->getName() );
        lineDetectionThread->shutdown();
        return;
    }
    prepareImage();

    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP( myCannyOutput, linesP, 1, CV_PI / 180, 70, 10, 50 ); // runs the actual detection
    // Draw the lines
    pthread_mutex_lock( &imageLock );
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[ i ];
        line( myLanesImage, Point( l[ 0 ], l[ 1 ] ), Point( l[ 2 ], l[ 3 ] ), Scalar( 0, 0, 255 ), 3, LINE_AA );
    }
    pthread_mutex_unlock( &imageLock );

}

void LineDetector::detectCars()
{
    if( abortS3 )
    {
        LogDebug( "Aborting %s.", carDetectionThread->getName() );
        carDetectionThread->shutdown();
        return;
    }

    if( not myGrayscaleImage.empty() )
    {
        vector<Rect> vehicle;
        myClassifier.detectMultiScale( myGrayscaleImage, vehicle );

        pthread_mutex_lock( &imageLock );
        for( size_t i = 0; i < vehicle.size(); ++i )
        {
            rectangle( myLanesImage, vehicle[ i ], CV_RGB( 255, 0, 0 ) );
        }
        pthread_mutex_unlock( &imageLock );
    }
}

bool LineDetector::loadClassifier( const String& classifier )
{
    if( not myClassifier.load( classifier ) )
    {
        LogFatal( "Failed to load classifier from %s.", classifier.c_str() );
        return false;
    }
    else
    {
        return true;
    }
}

void LineDetector::writeFrameToVideo()
{
    myVideoWriter.write( myVehiclesImage );
}

void LineDetector::createWindows()
{

    namedWindow( SOURCE_WINDOW_NAME, WINDOW_NORMAL );
    resizeWindow( SOURCE_WINDOW_NAME, Size( myFrameWidth, myFrameHeight ) );
    LogInfo( "Created window: %s", SOURCE_WINDOW_NAME.c_str() );

    namedWindow( DETECTED_LANES_IMAGE, WINDOW_NORMAL );
    resizeWindow( DETECTED_LANES_IMAGE, Size( myFrameWidth, myFrameHeight ) );
    LogInfo( "Created window: %s", DETECTED_LANES_IMAGE.c_str() );

    namedWindow( DETECTED_VEHICLES_IMAGE, WINDOW_NORMAL );
    resizeWindow( DETECTED_VEHICLES_IMAGE, Size( myFrameWidth, myFrameHeight ) );
    LogInfo( "Created window: %s", DETECTED_VEHICLES_IMAGE.c_str() );
}


void LineDetector::shutdown()
{
    LogDebug( "Shutting down threads!" );
    captureThread->shutdown();
    lineDetectionThread->shutdown();
    carDetectionThread->shutdown();
}

bool LineDetector::isCaptureThreadAlive()
{
    return captureThread->isThreadAlive();
}

bool LineDetector::isLineThreadAlive()
{
    return lineDetectionThread->isThreadAlive();
}

bool LineDetector::isCarThreadAlive()
{
    return carDetectionThread->isThreadAlive();
}

pthread_t LineDetector::getCaptureThreadId()
{
    return captureThread->getThreadId();
}