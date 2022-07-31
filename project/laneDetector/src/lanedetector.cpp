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

extern pthread_mutex_t cameraLock;
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
    pthread_mutex_lock( &cameraLock );
    myVideoCapture.read( mySource );
    myNewFrameReady = true;
    pthread_mutex_unlock( &cameraLock );
    LogDebug( "New image ready!" );
}

void* LineDetector::executeLine( void* context )
{
    ( ( LineDetector* )context )->detectLanes();
    return NULL;
}

void LineDetector::prepareImage()
{
    if( newFrameReady() )
    {
        pthread_mutex_lock( &cameraLock );
        myLanesImage = mySource.clone();
        cvtColor( mySource, myGrayscaleImage, COLOR_RGB2GRAY );
        GaussianBlur( mySource, tmp, Size( 5, 5 ), 0, 0, BORDER_DEFAULT );
        pthread_mutex_unlock( &cameraLock );
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
    vector<Rect> vehicle;
    myClassifier.detectMultiScale( myGrayscaleImage, vehicle );

    pthread_mutex_lock( &imageLock );
    for( size_t i = 0; i < vehicle.size(); ++i )
    {
        rectangle( myLanesImage, vehicle[ i ], CV_RGB( 255, 0, 0 ) );
    }
    pthread_mutex_unlock( &imageLock );

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



bool LineDetector::newFrameReady()
{
    return myNewFrameReady;
}

bool LineDetector::isFrameEmpty()
{
    return mySource.empty();
}

void LineDetector::showSourceImage()
{
    if( not mySource.empty() )
    {
        imshow( SOURCE_WINDOW_NAME, mySource );
    }
}

void LineDetector::showLanesImage()
{
    if( not myLanesImage.empty() )
    {
        imshow( DETECTED_LANES_IMAGE, myLanesImage );
    }
}

void LineDetector::showVehiclesImage()
{
    imshow( DETECTED_VEHICLES_IMAGE, myVehiclesImage );
}

void LineDetector::setHoughLinesPThreshold( int value )
{
    myHoughLinesPThreshold = value;
}

void LineDetector::setMinLineLength( int value )
{
    myMinLineLength = value;
}

void LineDetector::setMaxLineGap( int value )
{
    myMaxLineGap = value;
}

Mat LineDetector::getVehiclesImage()
{
    return myVehiclesImage;
}

int LineDetector::getFrameRate()
{
    return myVideoCapture.get( CAP_PROP_FPS );
}

int LineDetector::getFrameWidth()
{
    return myVideoCapture.get( CAP_PROP_FRAME_WIDTH );
}

int LineDetector::getFrameHeight()
{
    return myVideoCapture.get( CAP_PROP_FRAME_HEIGHT );
}


void LineDetector::shutdown()
{
    LogDebug( "Shutting down threads!" );
    captureThread->shutdown();
    lineDetectionThread->shutdown();
}

bool LineDetector::isAlive()
{
    return ( isCaptureThreadAlive() && isLineThreadAlive() );
}

bool LineDetector::isCaptureThreadAlive()
{
    return captureThread->isThreadAlive();
}

bool LineDetector::isLineThreadAlive()
{
    return lineDetectionThread->isThreadAlive();
}

pthread_t LineDetector::getCaptureThreadId()
{
    return captureThread->getThreadId();
}

sem_t* LineDetector::getCaptureSemaphore()
{
    return &captureThreadSem;
}

