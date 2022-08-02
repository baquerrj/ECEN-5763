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

#define RED    (cv::Scalar( 96,  94, 211))
#define BLUE   (cv::Scalar(203, 147, 114))

extern pthread_mutex_t rawBufferLock;
extern pthread_mutex_t grayscaleBufferLock;
extern pthread_mutex_t imageLock;

const cv::String LineDetector::SOURCE_WINDOW_NAME = "Source";
const cv::String LineDetector::DETECTED_LANES_IMAGE = "Detected Lanes (in red)";
const cv::String LineDetector::DETECTED_VEHICLES_IMAGE = "Detected Vehicles";
const cv::String LineDetector::CAR_CLASSIFIER = "cars.xml";

LineDetector::LineDetector( const ThreadConfigData* configData,
                            int deviceId,
                            const cv::String videoFilename,
                            bool writeOutputVideo,
                            const cv::String outputVideoFilename,
                            int frameWidth,
                            int frameHeight ) :
    foundLeft( false ),
    foundRight( false ),
    myHoughLinesPThreshold( INITIAL_PROBABILISTIC_HOUGH_THRESHOLD ),
    myMinLineLength( INITIAL_MAX_LINE_LAP ),
    myMaxLineGap( INITIAL_MIN_LINE_LENGTH ),
    myNewFrameReady( false ),
    myCreatedOk( true ),
    lanesReady( false ),
    carsReady( false ),
    myFrameWidth( frameWidth ),
    myFrameHeight( frameHeight ),
    myDeviceId( deviceId ),
    carsDeltaTimes( 0.0 ),
    lanesDeltaTimes( 0.0 ),
    annotationDeltaTimes( 0.0 ),
    framesProcessed( 0 ),
    carsThreadFrames( 0 ),
    lanesThreadFrames( 0 ),
    annotationThreadFrames( 0 ),
    myVideoFilename( videoFilename ),
    myOutputVideoFilename( outputVideoFilename ),
    carsStart( { 0,0 } ),
    carsStop( { 0,0 } ),
    lanesStart( { 0,0 } ),
    lanesStop( { 0,0 } ),
    annotationStart( { 0,0 } ),
    annotationStop( { 0,0 } )
{
    // carsReady = false;
    // lanesReady = false;
    // myCreatedOk = true;
    // myFrameHeight = frameHeight;
    // myFrameWidth = frameWidth;
    // myDeviceId = deviceId;
    // myVideoFilename = videoFilename;
    // framesProcessed = 0;
    // myHoughLinesPThreshold = INITIAL_PROBABILISTIC_HOUGH_THRESHOLD;
    // myMaxLineGap = INITIAL_MAX_LINE_LAP;
    // myMinLineLength = INITIAL_MIN_LINE_LENGTH;

    myVideoCapture = cv::VideoCapture( myVideoFilename );

    myVideoCapture.set( cv::CAP_PROP_FRAME_HEIGHT, ( double )frameHeight );
    myVideoCapture.set( cv::CAP_PROP_FRAME_WIDTH, ( double )frameWidth );

    roiPoints[ 0 ] = cv::Point( 350, 430 ); // top left
    roiPoints[ 1 ] = cv::Point( 770, 430 ); // top right
    roiPoints[ 2 ] = cv::Point( 770, 567 ); // bottom right
    roiPoints[ 3 ] = cv::Point( 350, 567 ); // bottom left
    // foundLeft = false;
    // foundRight = false;

    pthread_mutex_init( &lock, NULL );
    createWindows();

    p_myRawBuffer = new RingBuffer< cv::Mat >( RING_BUFFER_SIZE );
    p_myGrayscaleBuffer = new RingBuffer< cv::Mat >( RING_BUFFER_SIZE );
    p_myFinalBuffer = new RingBuffer< cv::Mat >( RING_BUFFER_SIZE );
    p_myReadyToAnnotateBuffer = new RingBuffer< cv::Mat >( RING_BUFFER_SIZE );

    leftPt1 = new RingBuffer < cv::Point >( 100 );
    leftPt2 = new RingBuffer < cv::Point >( 100 );
    rightPt1 = new RingBuffer < cv::Point >( 100 );
    rightPt2 = new RingBuffer < cv::Point >( 100 );
    vehicle = new RingBuffer < std::vector< cv::Rect > >( 100 );

    if( writeOutputVideo )
    {
        // myOutputVideoFilename = outputVideoFilename;
        if( myOutputVideoFilename.empty() or myOutputVideoFilename == "" )
        {
            myOutputVideoFilename = "output.avi";
        }
        myVideoWriter.open( myOutputVideoFilename,
                            cv::VideoWriter::fourcc( 'M', 'J', 'P', 'G' ),
                            getFrameRate(),
                            cv::Size( getFrameWidth(), getFrameHeight() ),
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
                if( myCreatedOk )
                {
                    ThreadConfigData annotationConfig = configData[ 3 ];
                    annotationThread = new CyclicThread( annotationConfig,
                                                           LineDetector::executeAnnotation,
                                                           this,
                                                           true );
                    if( NULL == annotationThread )
                    {
                        LogError( "Could not allocated memory for %s", annotationConfig.threadName.c_str() );
                        myCreatedOk = false;
                    }
                    if( not annotationThread->isThreadAlive() )
                    {
                        LogError( "Could not start thread for %s", annotationConfig.threadName.c_str() );
                        myCreatedOk = false;
                    }
                }
            }
        }
    }
}

LineDetector::~LineDetector()
{
    double lanesDeltaTimesMs = lanesDeltaTimes / lanesThreadFrames;
    double lanesDeltaTimeS = lanesDeltaTimesMs / 1000.0;
    printf( "**** LANE DETECTION ****\n\r" );
    printf( "Frames Processed: %ld\n\r", lanesThreadFrames );
    printf( "Average Lane Detection Frame Rate: %3.2f ms per frame\n\r", lanesDeltaTimesMs );
    printf( "Average Lane Detection Frame Rate: %3.2f frames per sec (fps)\n\r", 1.0 / lanesDeltaTimeS );
    printf( "**** LANE DETECTION ****\n\r" );

    pthread_mutex_destroy( &lock );
    if( myVideoCapture.isOpened() )
    {
        myVideoCapture.release();
    }

    if( myVideoWriter.isOpened() )
    {
        myVideoWriter.release();
    }

    cv::destroyAllWindows();

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

    if( annotationThread )
    {
        delete annotationThread;
        annotationThread = NULL;
    }

    if( p_myRawBuffer )
    {
        delete p_myRawBuffer;
        p_myRawBuffer = NULL;
    }

    if( p_myGrayscaleBuffer )
    {
        delete p_myGrayscaleBuffer;
        p_myGrayscaleBuffer = NULL;
    }

    if( p_myFinalBuffer )
    {
        delete p_myFinalBuffer;
        p_myFinalBuffer = NULL;
    }

    if( leftPt1 )
    {
        delete leftPt1;
        leftPt1 = NULL;
    }

    if( leftPt2 )
    {
        delete leftPt2;
        leftPt2 = NULL;
    }

    if( rightPt1 )
    {
        delete rightPt1;
        rightPt1 = NULL;
    }

    if( rightPt2 )
    {
        delete rightPt2;
        rightPt2 = NULL;
    }

    if( vehicle )
    {
        delete vehicle;
        vehicle = NULL;
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
        // LogDebug( "New raw image ready!" );
    }
}

void LineDetector::showLanesImage()
{
    if( not p_myFinalBuffer->isEmpty() )
    {
        // LogDebug( "Updating final image on window!" );
        imshow( DETECTED_LANES_IMAGE, p_myFinalBuffer->dequeue() );
    }
}

void* LineDetector::executeLine( void* context )
{
    ( ( LineDetector* )context )->detectLanes();
    return NULL;
}

void* LineDetector::executeAnnotation( void* context )
{
    ( ( LineDetector* )context )->annotateImage();
    return NULL;
}

void LineDetector::annotateImage()
{
    if( abortS4 )
    {
        LogDebug( "Aborting %s.", annotationThread->getName() );
        annotationThread->shutdown();
        return;
    }
    sem_wait( semS4 );
    // struct timespec now = { 0,0 };
    // clock_gettime( CLOCK_REALTIME, &now );
    // now.tv_nsec += ( 1000 * 100 ); // 100 usec wait
    // pthread_mutex_lock( &lock );

    if( rawImage.empty() )
    {
        pthread_mutex_unlock( &lock );
        return;
    }
    if( foundLeft )
    {
        // LogTrace( "Drawing detected left lane on image!" );
        if( not leftPt1->isEmpty() and not leftPt2->isEmpty() )
        {
            line( rawImage, leftPt1->dequeue(), leftPt2->dequeue(), RED, 2, cv::LINE_4 );
        }
    }

    // if( foundRight and isInsideRoi( rightPt1 ) and isInsideRoi( rightPt2 ) )
    if( foundRight )
    {
        // LogTrace( "Drawing detected right lane on image!" );
        if( not rightPt1->isEmpty() and not rightPt2->isEmpty() )
        {
            line( rawImage, rightPt1->dequeue(), rightPt2->dequeue(), RED, 2, cv::LINE_4 );
        }
    }

    // LogTrace( "Drawing detected ROI on image!" );
    rectangle( rawImage, roiPoints[ 0 ], roiPoints[ 2 ], BLUE, 1, cv::LINE_AA );
    putText( rawImage, "ROI", roiPoints[ 0 ], cv::FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1 );

    std::vector< cv::Rect > tmpVehicle;
    if( not vehicle->isEmpty() )
    {
        tmpVehicle = vehicle->dequeue();
    }
    for( size_t i = 0; i < tmpVehicle.size(); ++i )
    {
        rectangle( rawImage, tmpVehicle[ i ], CV_RGB( 255, 0, 0 ) );
    }

    imshow( DETECTED_LANES_IMAGE, rawImage );
    // pthread_mutex_unlock( &lock );
    // while( p_myFinalBuffer->isFull() and not abortS4 )
    // {
//     usleep( int( 1000 * 2 ) );
    //     continue;
    // }

    // if( not abortS4 )
    // {
    //     p_myFinalBuffer->enqueue( rawImage );
    //     LogDebug( "New processed image ready!" );
        framesProcessed++;
    // }

    // showLanesImage();
}

void* LineDetector::executeCar( void* context )
{
    ( ( LineDetector* )context )->detectCars();
    return NULL;
}



void LineDetector::detectLanes()
{
    // LogTrace( "Entered" );
    if( abortS2 )
    {
        LogDebug( "Aborting %s.", lineDetectionThread->getName() );
        lineDetectionThread->shutdown();
        return;
    }
    sem_wait( semS2 );
    clock_gettime(CLOCK_REALTIME, &lanesStart );
    pthread_mutex_lock( &rawBufferLock );
    if( not p_myRawBuffer->isEmpty() )
    {
        rawImage = p_myRawBuffer->dequeue();
    }
    else
    {
        pthread_mutex_unlock( &rawBufferLock );
        // LogTrace( "Raw Buffer is empty! Looping around...");
        return;
    }
    pthread_mutex_unlock( &rawBufferLock );

    cv::cvtColor( rawImage, myGrayscaleImage, cv::COLOR_BGR2GRAY );

    pthread_mutex_lock( &grayscaleBufferLock );
    if( not p_myGrayscaleBuffer->isFull() )
    {
        p_myGrayscaleBuffer->enqueue( myGrayscaleImage );
    }
    pthread_mutex_unlock( &grayscaleBufferLock );

    roi = myGrayscaleImage( cv::Rect(roiPoints[0], roiPoints[2] ) );

    cv::medianBlur( roi, roi, 5 );

    cv::adaptiveThreshold( roi, roi, 255,
                           cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, -2 );

    cv::Vec4i left;
    cv::Vec4i right;

    foundLeft = false;
    foundRight = false;

    findLeftLane(left);
    findRightLane(right);

    // // pthread_mutex_lock( &imageLock );
    // // if( foundLeft and isInsideRoi( leftPt1 ) and isInsideRoi( leftPt2 ) )
    // if( foundLeft )
    // {
    //     LogTrace( "Drawing detected left lane on image!" );
    //     line(rawImage, leftPt1, leftPt2, Scalar(255, 0, 0), 3, LINE_4 );
    // }

    // // if( foundRight and isInsideRoi( rightPt1 ) and isInsideRoi( rightPt2 ) )
    // if( foundRight )
    // {
    //     LogTrace( "Drawing detected right lane on image!" );
    //     line(rawImage, rightPt1, rightPt2, Scalar(255, 0, 0), 3, LINE_4 );
    // }

    // LogTrace( "Drawing detected ROI on image!" );
    // cv::Rectangle( rawImage, roiPoints[ 0 ], roiPoints[ 2 ], Scalar( 255, 0, 0 ), 2, LINE_AA );
    // putText( rawImage, "ROI", roiPoints[ 0 ], FONT_HERSHEY_SIMPLEX, 0.5, Scalar( 255, 0, 0 ), 1.5 );

    // while( p_myFinalBuffer->isFull() and not abortS2 )
    // {
    //     usleep( int(1000 * 2));
    //     continue;
    // }
    // if( not abortS2 )
    // {
    //     p_myFinalBuffer->enqueue( rawImage );
    //     LogDebug( "New processed image ready!" );
    //     framesProcessed++;
    // }
    // pthread_mutex_unlock( &imageLock );
    // LogTrace( "Exiting" );
    clock_gettime( CLOCK_REALTIME, &lanesStop );
    lanesThreadFrames++;
    lanesDeltaTimes += delta_t( &lanesStop, &lanesStart );
}

void LineDetector::findLeftLane( cv::Vec4i left )
{
    vector< cv::Vec3f > lines;
    uint32_t i = 0;

    cv::HoughLines(
        roi,
        lines,
        1,
        CV_PI / 180,
        30,
        0,
        0,
        // 0.174533,
        // 1.134464
        (10)*(CV_PI/180),
        (65)*(CV_PI/180)
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
        cv::Point2f ret;
        cv::Point2f pt1 = cv::Point2f( left[ 0 ], left[ 1 ] );
        cv::Point2f pt2 = cv::Point2f( left[ 2 ], left[ 3 ] );

        if( intersection( cv::Point( 0, 0 ), cv::Point( 1, 0 ), pt1, pt2, ret ) )
        {
            if( not leftPt1->isFull() )
            {
                leftPt1->enqueue( cv::Point( std::round( ret.x ), std::round( ret.y ) ) + roiPoints[ 0 ] );
            }
        }
        else
        {
            foundLeft = false;
        }
        if( intersection( cv::Point( 0, roi.rows - 1 ), cv::Point( 1, roi.rows - 1 ), pt1, pt2, ret ) )
        {
            if( not leftPt2->isFull() )
            {
                leftPt2->enqueue( cv::Point( round( ret.x ), round( ret.y ) ) + roiPoints[ 0 ] );
            }
        }
        else
        {
            foundLeft = false;
        }
    }
}

void LineDetector::findRightLane( cv::Vec4i right )
{

    vector< cv::Vec3f > lines;
    uint32_t i = 0;
    cv::HoughLines(
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

        cv::Point2f ret;
        cv::Point2f pt1 = cv::Point2f( right[ 0 ], right[ 1 ] );
        cv::Point2f pt2 = cv::Point2f( right[ 2 ], right[ 3 ] );

        if( intersection( cv::Point( 0, 0 ), cv::Point( 1, 0 ), pt1, pt2, ret ) )
        {
            if( not rightPt1->isFull() )
            {
                rightPt1->enqueue( cv::Point( ret ) + roiPoints[ 0 ] );
            }
        }
        else
        {
            foundRight = false;
        }

        if( intersection( cv::Point( 0, roi.rows - 1 ), cv::Point( 1, roi.rows - 1 ), pt1, pt2, ret ) )
        {
            if( not rightPt2->isFull() )
            {
                rightPt2->enqueue( cv::Point( ret ) + roiPoints[ 0 ] );
            }
        }
        else
        {
            foundRight = false;
        }
    }

}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
// https://stackoverflow.com/questions/7446126/opencv-2d-line-intersection-helper-function
bool LineDetector::intersection( cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2,
                                 cv::Point2f& r )
{
    cv::Point2f x = o2 - o1;
    cv::Point2f d1 = p1 - o1;
    cv::Point2f d2 = p2 - o2;

    float cross = d1.x * d2.y - d1.y * d2.x;
    if( std::abs( cross ) < 1e-8 )
    {
        return false;
    }
    else
    {
        double t1 = ( x.x * d2.y - x.y * d2.x ) / cross;
        r = o1 + d1 * t1;
        return true;
    }
}

void LineDetector::detectCars()
{
    if( abortS3 )
    {
        LogDebug( "Aborting %s.", carDetectionThread->getName() );
        carDetectionThread->shutdown();
        return;
    }
    sem_wait( semS3 );

    pthread_mutex_lock( &grayscaleBufferLock );
    if( not p_myGrayscaleBuffer->isEmpty() )
    {
        imageToProcess = p_myGrayscaleBuffer->dequeue();
    }
    pthread_mutex_unlock( &grayscaleBufferLock );

    std::vector< cv::Rect > tmpVehicle;
    myClassifier.detectMultiScale( imageToProcess, tmpVehicle );
    if( not vehicle->isFull() )
    {
        vehicle->enqueue( tmpVehicle );
    }

    // pthread_mutex_lock( &imageLock );
    // for( size_t i = 0; i < vehicle.size(); ++i )
    // {
    //     rectangle( myLanesImage, vehicle[ i ], CV_RGB( 255, 0, 0 ) );
    // }
    // pthread_mutex_unlock( &imageLock );
}

bool LineDetector::loadClassifier( const cv::String& classifier )
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

    cv::namedWindow( DETECTED_LANES_IMAGE, cv::WINDOW_NORMAL );
    cv::resizeWindow( DETECTED_LANES_IMAGE, cv::Size( myFrameWidth, myFrameHeight ) );
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
    annotationThread->shutdown();
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

bool LineDetector::isAnnotationThreadAlive()
{
    return annotationThread->isThreadAlive();
}

pthread_t LineDetector::getCaptureThreadId()
{
    return captureThread->getThreadId();
}