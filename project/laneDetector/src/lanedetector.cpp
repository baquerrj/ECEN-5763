/**
 * @file houghclines.cpp
 * @brief This program demonstrates line finding with the Hough transform
 */

#include "lanedetector.hpp"
#include "configuration.hpp"
#include "thread.hpp"
#include "thread_utils.hpp"
#include "logging.hpp"
#include "string.h"
using namespace std;

extern pthread_mutex_t cameraLock;

const String LineDetector::SOURCE_WINDOW_NAME = "Source";
const String LineDetector::DETECTED_LANES_IMAGE = "Detected Lanes (in red)";
const String LineDetector::DETECTED_VEHICLES_IMAGE = "Detected Vehicles";
const String LineDetector::CAR_CLASSIFIER = "cars.xml";

LineDetector::LineDetector( const ThreadConfigData configData,
                  int deviceId,
                  const String videoFilename,
                  bool writeOutputVideo,
                  const String outputVideoFilename,
                  int frameWidth,
                  int frameHeight )
{
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
    thread = new CyclicThread( configData,
                                LineDetector::execute,
                                this,
                                true );
    if( NULL == thread )
    {
        logging::ERROR( "Could not allocated memory for " + configData.threadName );
        exit( -1 );
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

    if( thread )
    {
        delete thread;
    }
}

void* LineDetector::execute( void* context )
{
    ( (LineDetector*)context )->readFrame();
    return NULL;
}

void LineDetector::prepareImage()
{
    cvtColor( mySource, myGrayscaleImage, COLOR_RGB2GRAY );
    GaussianBlur( mySource, tmp, Size( 5, 5 ), 0, 0, BORDER_DEFAULT );
    Canny( tmp, myCannyOutput, 40, 120, 3, true );
    cvtColor( myCannyOutput, myLanesImage, COLOR_GRAY2RGB );
    cvtColor( myCannyOutput, myVehiclesImage, COLOR_GRAY2RGB );
}

void LineDetector::detectLanes()
{
    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP( myCannyOutput, linesP, 1, CV_PI / 180, 70, 10, 50 ); // runs the actual detection
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[ i ];
        line( myLanesImage, Point( l[ 0 ], l[ 1 ] ), Point( l[ 2 ], l[ 3 ] ), Scalar( 0, 0, 255 ), 3, LINE_AA );
    }

    return;
}

void LineDetector::detectCars()
{
    vector<Rect> vehicle;
    myClassifier.detectMultiScale( myGrayscaleImage, vehicle );

    for( size_t i = 0; i < vehicle.size(); ++i )
    {
        rectangle( myVehiclesImage, vehicle[ i ], CV_RGB( 255, 0, 0 ) );
    }
}

bool LineDetector::loadClassifier( const String& classifier )
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

void LineDetector::writeFrameToVideo()
{
    myVideoWriter.write( myVehiclesImage );
}

void LineDetector::createWindows()
{

    namedWindow( SOURCE_WINDOW_NAME, WINDOW_NORMAL );
    resizeWindow( SOURCE_WINDOW_NAME, Size( myFrameWidth, myFrameHeight ) );
    printf( "Created window: %s\n\r", SOURCE_WINDOW_NAME.c_str() );

    namedWindow( DETECTED_LANES_IMAGE, WINDOW_NORMAL );
    resizeWindow( DETECTED_LANES_IMAGE, Size( myFrameWidth, myFrameHeight ) );
    printf( "Created window: %s\n\r", DETECTED_LANES_IMAGE.c_str() );

    namedWindow( DETECTED_VEHICLES_IMAGE, WINDOW_NORMAL );
    resizeWindow( DETECTED_VEHICLES_IMAGE, Size( myFrameWidth, myFrameHeight ) );
    printf( "Created window: %s\n\r", DETECTED_VEHICLES_IMAGE.c_str() );
}

void LineDetector::readFrame()
{
    while( abortS1 )
    {
        thread->shutdown();
        return;
    }

    pthread_mutex_lock( &cameraLock );
    myVideoCapture.read( mySource );
    myNewFrameReady = true;
    pthread_mutex_unlock( &cameraLock );
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
    imshow( SOURCE_WINDOW_NAME, mySource );
}

void LineDetector::showLanesImage()
{
    imshow( DETECTED_LANES_IMAGE, myLanesImage );
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
    thread->shutdown();
}
bool LineDetector::isAlive()
{
    return alive;
}
bool LineDetector::isThreadAlive()
{
    return thread->isThreadAlive();
}
pthread_t LineDetector::getThreadId()
{
    return thread->getThreadId();
}
sem_t* LineDetector::getSemaphore()
{
    return &sem;
}

