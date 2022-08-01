/**
 * @file houghclines.cpp
 * @brief This program demonstrates line finding with the Hough transform
 */

#include "lanedetector.hpp"
#include "configuration.hpp"
#include "thread.hpp"
#include "thread_utils.hpp"
#include "string.h"
#include <unistd.h>

using namespace std;

#include "Logger.h"
#include "RingBuffer.h"
extern pthread_mutex_t rawBufferLock;
extern pthread_mutex_t grayscaleBufferLock;
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
    framesProcessed = 0;
    myHoughLinesPThreshold = INITIAL_PROBABILISTIC_HOUGH_THRESHOLD;
    myMaxLineGap = INITIAL_MAX_LINE_LAP;
    myMinLineLength = INITIAL_MIN_LINE_LENGTH;

    myVideoCapture = VideoCapture( myVideoFilename );

    myVideoCapture.set( CAP_PROP_FRAME_HEIGHT, ( double )frameHeight );
    myVideoCapture.set( CAP_PROP_FRAME_WIDTH, ( double )frameWidth );

    roiPoints[ 0 ] = Point( 350, 430 ); // top left
    roiPoints[ 1 ] = Point( 750, 430 ); // top right
    roiPoints[ 2 ] = Point( 750, 567 ); // bottom right
    roiPoints[ 3 ] = Point( 350, 567 ); // bottom left
    foundLeft = false;
    foundRight = false;

    createWindows();

    p_myRawBuffer = new RingBuffer< Mat >( RING_BUFFER_SIZE );
    p_myGrayscaleBuffer = new RingBuffer< Mat >( RING_BUFFER_SIZE );
    p_myFinalBuffer = new RingBuffer< Mat >( RING_BUFFER_SIZE );

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
    sem_wait( semS1 );
    myNewFrameReady = false;
    pthread_mutex_lock( &rawBufferLock );
    if( p_myRawBuffer->isFull() )
    {
        myNewFrameReady = false;
        pthread_mutex_unlock( &rawBufferLock );
    }
    else
    {
        myVideoCapture.read( tmp );
        p_myRawBuffer->enqueue( tmp );
        pthread_mutex_unlock( &rawBufferLock );
        myNewFrameReady = true;
        LogDebug( "New raw image ready!" );
    }
}

void LineDetector::showLanesImage()
{
    if( not p_myFinalBuffer->isEmpty() )
    {
        LogDebug( "Updating final image on window!" );
        imshow( DETECTED_LANES_IMAGE, p_myFinalBuffer->dequeue() );
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

#define RED    (Scalar( 96,  94, 211))
#define BLUE   (Scalar(203, 147, 114))

void LineDetector::detectLanes()
{
    LogTrace( "Entered" );
    if( abortS2 )
    {
        LogDebug( "Aborting %s.", lineDetectionThread->getName() );
        lineDetectionThread->shutdown();
        return;
    }
    sem_wait( semS2 );

    pthread_mutex_lock( &rawBufferLock );
    if( not p_myRawBuffer->isEmpty() )
    {
        rawImage = p_myRawBuffer->dequeue();
    }
    else
    {
        pthread_mutex_unlock( &rawBufferLock );
        LogTrace( "Raw Buffer is empty! Looping around...");
        return;
    }
    pthread_mutex_unlock( &rawBufferLock );

    annot = Mat(rawImage);
    cvtColor( rawImage, myGrayscaleImage, COLOR_BGR2GRAY );

    // pthread_mutex_lock( &grayscaleBufferLock );
    // if( not p_myGrayscaleBuffer->isFull() )
    // {
    //     p_myGrayscaleBuffer->enqueue( myGrayscaleImage );
    // }
    // pthread_mutex_unlock( &grayscaleBufferLock );

    roi = myGrayscaleImage( Rect(roiPoints[0], roiPoints[2] ) );

    medianBlur( roi, roi, 5 );

    adaptiveThreshold( roi, roi, 255,
    ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, -2 );

    Vec4i left;
    Vec4i right;

    foundLeft = false;
    foundRight = false;

    findLeftLane(left);
    findRightLane(right);

    // pthread_mutex_lock( &imageLock );
    // if( foundLeft and isInsideRoi( leftPt1 ) and isInsideRoi( leftPt2 ) )
    if( foundLeft )
    {
        LogDebug( "Drawing detected left lane on image!" );
        line(rawImage, leftPt1, leftPt2, Scalar(255, 0, 0), 3, LINE_4 );
    }

    // if( foundRight and isInsideRoi( rightPt1 ) and isInsideRoi( rightPt2 ) )
    if( foundRight )
    {
        LogDebug( "Drawing detected right lane on image!" );
        line(rawImage, rightPt1, rightPt2, Scalar(255, 0, 0), 3, LINE_4 );
    }

    LogDebug( "Drawing detected ROI on image!" );
    rectangle( rawImage, roiPoints[ 0 ], roiPoints[ 2 ], Scalar( 255, 0, 0 ), 2, LINE_AA );
    putText( rawImage, "ROI", roiPoints[ 0 ], FONT_HERSHEY_SIMPLEX, 0.5, Scalar( 255, 0, 0 ), 1.5 );

    while( p_myFinalBuffer->isFull() and not abortS2 )
    {
        usleep( int(1000 * 2));
        continue;
    }
    if( not abortS2 )
    {
        p_myFinalBuffer->enqueue( rawImage );
        LogDebug( "New processed image ready!" );
        framesProcessed++;
    }
    // pthread_mutex_unlock( &imageLock );
    LogTrace( "Exiting" );
}

void LineDetector::findLeftLane( Vec4i left )
{
    vector< Vec3f > lines;
    uint32_t i = 0;

    HoughLines(
        roi,
        lines,
        1,
        CV_PI / 180,
        30,
        0,
        0,
        0.174533,
        1.134464
    );

    while( !( foundLeft ) && i < lines.size() )
    {

        // sourced from OpenCV Hough tutorial:
        float rho = lines[ i ][ 0 ], theta = lines[ i ][ 1 ];
        if( abs( rho ) > 90 && abs( rho ) < 150 )
        {
            //LOGP("rho: %f, theta: %f, votes: %f\n", rho, theta*180/CV_PI, lines[i][2]);
            double a = cos( theta ), b = sin( theta );
            double x0 = a * rho, y0 = b * rho;
            left[ 0 ] = cvRound( x0 + 1000 * ( -b ) );
            left[ 1 ] = cvRound( y0 + 1000 * ( a ) );
            left[ 2 ] = cvRound( x0 - 1000 * ( -b ) );
            left[ 3 ] = cvRound( y0 - 1000 * ( a ) );
            foundLeft = true;
        }

        i++;
    }
    if( foundLeft )
    {
        Point2f ret;
        Point2f pt1 = Point2f( left[ 0 ], left[ 1 ] );
        Point2f pt2 = Point2f( left[ 2 ], left[ 3 ] );

        if( getIntersection( Point( 0, 0 ), Point( 1, 1 ), pt1, pt2, ret ) )
        {
            leftPt1 = Point( round( ret.x ), round( ret.y ) ) + roiPoints[ 0 ];
        }
        else
        {
            foundLeft = false;
        }
        if( getIntersection( Point( 0, roi.rows - 1 ), Point( 1, roi.rows - 1 ), pt1, pt2, ret ) )
        {
            leftPt2 = Point( round( ret.x ), round( ret.y ) ) + roiPoints[ 0 ];
        }
        else
        {
            foundLeft = false;
        }
    }
}

void LineDetector::findRightLane( Vec4i right )
{

    vector< Vec3f > lines;
    uint32_t i = 0;
    HoughLines(
        roi,           // image
        lines,         // lines
        1,             // rho resolution of accumulator in pixels
        CV_PI / 180,     // theta resolution of accumulator
        30,    // accumulator threshold, only lines >threshold returned
        0,             // srn - set to 0 for classical Hough
        0,             // stn - set to 0 for classical Hough
        2.007129,      // minimum theta
        2.967060       // maximum theta
    );


    while( !( foundRight ) && i < lines.size() )
    {

        // sourced from OpenCV Hough tutorial:
        float rho = lines[ i ][ 0 ], theta = lines[ i ][ 1 ];
        if( abs( rho ) > 150 && abs( rho ) < 300 )
        {
            //LOGP("rho: %f, theta: %f, votes: %f\n", rho, theta*180/CV_PI, lines[i][2]);
            double a = cos( theta ), b = sin( theta );
            double x0 = a * rho, y0 = b * rho;
            right[ 0 ] = cvRound( x0 + 1000 * ( -b ) );
            right[ 1 ] = cvRound( y0 + 1000 * ( a ) );
            right[ 2 ] = cvRound( x0 - 1000 * ( -b ) );
            right[ 3 ] = cvRound( y0 - 1000 * ( a ) );
            foundRight = true;
        }

        i++;
    }



    if( foundRight )
    {

        Point2f ret;
        Point2f pt1 = Point2f( right[ 0 ], right[ 1 ] );
        Point2f pt2 = Point2f( right[ 2 ], right[ 3 ] );

        if( getIntersection( Point( 0, 0 ), Point( 1, 0 ), pt1, pt2, ret ) )
        {
            rightPt1 = Point( ret ) + roiPoints[ 0 ];
        }
        else
        {
            foundRight = false;
        }

        if( getIntersection( Point( 0, roi.rows - 1 ), Point( 1, roi.rows - 1 ), pt1, pt2, ret ) )
        {
            rightPt2 = Point( ret ) + roiPoints[ 0 ];
        }
        else
        {
            foundRight = false;
        }
    }

}

bool LineDetector::isInsideRoi( Point p )
{
    return ( 0 <= p.x and p.x < annot.cols and 0 <= p.y and p.y < annot.rows );
}

bool LineDetector::getIntersection( Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r )
{
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x * d2.y - d1.y * d2.x;
    if( abs( cross ) < /*EPS*/1e-8 )
        return false;

    double t1 = ( x.x * d2.y - x.y * d2.x ) / cross;
    r = o1 + d1 * t1;
    return true;
}

void LineDetector::detectCars()
{
    if( abortS3 )
    {
        LogDebug( "Aborting %s.", carDetectionThread->getName() );
        carDetectionThread->shutdown();
        return;
    }

    pthread_mutex_lock( &grayscaleBufferLock );
    if( not p_myGrayscaleBuffer->isEmpty() )
    {
        imageToProcess = p_myGrayscaleBuffer->dequeue();
    }
    pthread_mutex_unlock( &grayscaleBufferLock );

    vector<Rect> vehicle;
    myClassifier.detectMultiScale( imageToProcess, vehicle );

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
    // namedWindow( SOURCE_WINDOW_NAME, WINDOW_NORMAL );
    // resizeWindow( SOURCE_WINDOW_NAME, Size( myFrameWidth, myFrameHeight ) );
    // LogInfo( "Created window: %s", SOURCE_WINDOW_NAME.c_str() );

    namedWindow( DETECTED_LANES_IMAGE, WINDOW_NORMAL );
    resizeWindow( DETECTED_LANES_IMAGE, Size( myFrameWidth, myFrameHeight ) );
    LogInfo( "Created window: %s", DETECTED_LANES_IMAGE.c_str() );

    // namedWindow( DETECTED_VEHICLES_IMAGE, WINDOW_NORMAL );
    // resizeWindow( DETECTED_VEHICLES_IMAGE, Size( myFrameWidth, myFrameHeight ) );
    // LogInfo( "Created window: %s", DETECTED_VEHICLES_IMAGE.c_str() );
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