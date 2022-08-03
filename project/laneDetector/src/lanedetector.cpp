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
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

#include "Logger.h"
#include "RingBuffer.h"

#define RED    (cv::Scalar( 96,  94, 211))
#define BLUE   (cv::Scalar(203, 147, 114))

extern pthread_mutex_t rawBufferLock;
extern pthread_mutex_t carRingLock;
extern pthread_mutex_t framesProcessedLock;

const cv::String LineDetector::SOURCE_WINDOW_NAME = "Source";
const cv::String LineDetector::DETECTED_LANES_IMAGE = "Detected Lanes (in red)";
const cv::String LineDetector::DETECTED_VEHICLES_IMAGE = "Detected Vehicles";
const cv::String LineDetector::CAR_CLASSIFIER = "cars.xml";


LineDetector::LineDetector( const ThreadConfigData* configData,
                            int deviceId,
                            const cv::String videoFilename,
                            bool saveFrames,
                            const cv::String outputDirectory,
                            bool showWindows,
                            int frameWidth,
                            int frameHeight ) :
    foundLeft( false ),
    foundRight( false ),
    myCreatedOk( true ),
    lanesReady( false ),
    carsReady( false ),
    myFrameWidth( frameWidth ),
    myFrameHeight( frameHeight ),
    myDeviceId( deviceId ),
    showWindows( showWindows ),
    carsDeltaTimes( 0.0 ),
    lanesDeltaTimes( 0.0 ),
    annotationDeltaTimes( 0.0 ),
    carsThreadFrames( 0 ),
    lanesThreadFrames( 0 ),
    annotationThreadFrames( 0 ),
    myVideoFilename( videoFilename ),
    myOutputDirectory( outputDirectory ),
    carsStart( { 0,0 } ),
    carsStop( { 0,0 } ),
    lanesStart( { 0,0 } ),
    lanesStop( { 0,0 } ),
    annotationStart( { 0,0 } ),
    annotationStop( { 0,0 } ),
    numberOfEmptyFrames( 0 )
{
    myVideoCapture = cv::VideoCapture( myVideoFilename );

    myVideoCapture.set( cv::CAP_PROP_FRAME_HEIGHT, ( double )frameHeight );
    myVideoCapture.set( cv::CAP_PROP_FRAME_WIDTH, ( double )frameWidth );

    roiPoints[ 0 ] = cv::Point( 350, 430 ); // top left
    roiPoints[ 1 ] = cv::Point( 770, 430 ); // top right
    roiPoints[ 2 ] = cv::Point( 770, 567 ); // bottom right
    roiPoints[ 3 ] = cv::Point( 350, 567 ); // bottom left

    carPoints[ 0 ] = cv::Point( 0, 380 ); // top left
    carPoints[ 1 ] = cv::Point( 800, 380 ); // top right
    carPoints[ 2 ] = cv::Point( 800, 567 ); // bottom right
    carPoints[ 3 ] = cv::Point( 0, 567 ); // bottom left

    pthread_mutex_init( &lock, NULL );
    pthread_mutex_init( &frameLock, NULL );
    pthread_mutex_init( &roiLock, NULL );
    if( showWindows )
    {
        createWindows();
    }

    p_myRawBuffer = new RingBuffer< cv::Mat >( RING_BUFFER_SIZE );
    p_bufferForCarDetection = new RingBuffer< cv::Mat >( RING_BUFFER_SIZE );
    p_myReadyToAnnotateBuffer = new RingBuffer< cv::Mat >( RING_BUFFER_SIZE );

    p_myFinalBuffer = new RingBuffer< frame_s >( RING_BUFFER_SIZE );
    frames = new RingBuffer < frame_s >( 100 );

    if( saveFrames )
    {
        // myOutputDirectory = outputDirectory;
        if( myOutputDirectory.empty() or myOutputDirectory == "" )
        {
            myOutputDirectory = "outputFrames/";
        }

        mkdir( myOutputDirectory.c_str(), 0777 );
        // myVideoWriter.open( myOutputDirectory,
        //                     cv::VideoWriter::fourcc( 'M', 'J', 'P', 'G' ),
        //                     getFrameRate(),
        //                     cv::Size( getFrameWidth(), getFrameHeight() ),
        //                     true );
    }


    if( not loadClassifier( LineDetector::CAR_CLASSIFIER ) )
    {
        LogFatal( "Unable to load Haar classifier!" );
        myCreatedOk = false;
    }

    if( myCreatedOk )
    {
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

void LineDetector::printFrameRates()
{
    double lanesDeltaTimesMs = lanesDeltaTimes / lanesThreadFrames;
    double lanesDeltaTimeS = lanesDeltaTimesMs / 1000.0;
    printf( "**** LANE DETECTION ****\n\r" );
    printf( "Frames Processed: %ld\n\r", lanesThreadFrames );
    printf( "Average Lane Detection Frame Rate: %3.2f ms per frame\n\r", lanesDeltaTimesMs );
    printf( "Average Lane Detection Frame Rate: %3.2f frames per sec (fps)\n\r", 1.0 / lanesDeltaTimeS );
    printf( "**** LANE DETECTION ****\n\r" );

    double carsDeltaTimesMs = carsDeltaTimes / carsThreadFrames;
    double carsDeltaTimeS = carsDeltaTimesMs / 1000.0;
    printf( "**** CAR DETECTION ****\n\r" );
    printf( "Frames Processed: %ld\n\r", carsThreadFrames );
    printf( "Average Lane Detection Frame Rate: %3.2f ms per frame\n\r", carsDeltaTimesMs );
    printf( "Average Lane Detection Frame Rate: %3.2f frames per sec (fps)\n\r", 1.0 / carsDeltaTimeS );
    printf( "**** CAR DETECTION ****\n\r" );
}

LineDetector::~LineDetector()
{


    pthread_mutex_destroy( &lock );
    pthread_mutex_destroy( &roiLock );
    pthread_mutex_destroy( &frameLock );
    if( myVideoCapture.isOpened() )
    {
        myVideoCapture.release();
    }

    // if( myVideoWriter.isOpened() )
    // {
    //     myVideoWriter.release();
    // }

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

    if( p_bufferForCarDetection )
    {
        delete p_bufferForCarDetection;
        p_bufferForCarDetection = NULL;
    }

    if( p_myFinalBuffer )
    {
        delete p_myFinalBuffer;
        p_myFinalBuffer = NULL;
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
    if( numberOfEmptyFrames >= EMPTY_FRAMES_PERSISTENCY_CHECK )
    {
        abortS1 = true;
        abortS2 = true;
        abortS3 = true;
        abortS4 = true;
        abortSequencer = true;
        return;
    }
    sem_wait( semS1 );
    while( p_myRawBuffer->isFull() )
    {
        LogTrace( "Raw Buffer is full" );
        usleep( 1 );
    }
    myVideoCapture.read( tmp );
    if( tmp.empty() )
    {
        numberOfEmptyFrames++;
    }
    p_myRawBuffer->enqueue( tmp );
}

void LineDetector::showLanesImage()
{
    if( p_myFinalBuffer->isEmpty() )
    {
        return;
    }
    frame_s f = p_myFinalBuffer->dequeue();
    cv::Mat image = f.currentAnnotatedImage;
    if( showWindows )
    {
        cv::imshow( DETECTED_LANES_IMAGE, image );
    }

    if( saveFrames )
    {
        char filepath[ 100 ];
        sprintf( filepath, "%s/%08ld.jpg", myOutputDirectory.c_str(), f.number );
        LogTrace( "writing %s", filepath );
        cv::imwrite( std::string( filepath ), image );
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

    if( frames->isEmpty() )
    {
        return;
    }
    pthread_mutex_lock( &frameLock );
    frame_s f = frames->dequeue();
    pthread_mutex_unlock( &frameLock );
    f.currentAnnotatedImage = f.currentRawImage.clone();

    if( carDetectionThread->isThreadAlive() and f.vehicle.size() != 0 )
    {
        LogTrace( "Annotating cars on image (frame #%ld)!", framesProcessed );
        for( size_t i = 0; i < f.vehicle.size(); ++i )
        {
            cv::Point rect[ 2 ];
            rect[ 0 ].x = f.vehicle[ i ].x;
            rect[ 0 ].y = f.vehicle[ i ].y + 380;
            rect[ 1 ].x = f.vehicle[ i ].x + f.vehicle[ i ].width;
            rect[ 1 ].y = f.vehicle[ i ].y + f.vehicle[ i ].height + 380;
            rectangle( f.currentAnnotatedImage, rect[ 0 ], rect[ 1 ], CV_RGB( 255, 0, 0 ), 3 );
        }
    }

    if( foundLeft )
    {
        line( f.currentAnnotatedImage, f.leftPt1, f.leftPt2, RED, 2, cv::LINE_4 );
    }

    if( foundRight )
    {
        line( f.currentAnnotatedImage, f.rightPt1, f.rightPt2, RED, 2, cv::LINE_4 );
    }

    rectangle( f.currentAnnotatedImage, roiPoints[ 0 ], roiPoints[ 2 ], BLUE, 1, cv::LINE_AA );
    putText( f.currentAnnotatedImage, "ROI", roiPoints[ 0 ], cv::FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1 );
    rectangle( f.currentAnnotatedImage, carPoints[ 0 ], carPoints[ 2 ], BLUE, 2, cv::LINE_AA );
    pthread_mutex_lock( &framesProcessedLock );
    framesProcessed++;
    f.number = framesProcessed;
    pthread_mutex_unlock( &framesProcessedLock );
    while( p_myFinalBuffer->isFull() )
    {
        usleep( 1 );
    }
    p_myFinalBuffer->enqueue( f );
}

void* LineDetector::executeCar( void* context )
{
    ( ( LineDetector* )context )->detectCars();
    return NULL;
}



void LineDetector::detectLanes()
{
    if( abortS2 )
    {
        LogDebug( "Aborting %s.", lineDetectionThread->getName() );
        lineDetectionThread->shutdown();
        return;
    }
    sem_wait( semS2 );
    clock_gettime( CLOCK_REALTIME, &lanesStart );
    if( p_myRawBuffer->isEmpty() )
    {
        return;
    }
    cv::Mat raw = p_myRawBuffer->dequeue();

    if( raw.empty() )
    {
        return;
    }

    pthread_mutex_lock( &roiLock );
    roi = raw( cv::Rect( roiPoints[ 0 ], roiPoints[ 2 ] ) );

    cv::cvtColor( roi, roi, cv::COLOR_BGR2GRAY );


    cv::medianBlur( roi, roi, 5 );

    cv::adaptiveThreshold( roi, roi, 255,
                           cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, -2 );

    frame_s f;
    cv::Vec4i left;
    cv::Vec4i right;

    foundLeft = false;
    foundRight = false;

    pthread_mutex_lock( &frameLock );
    findLeftLane( left, f );
    findRightLane( right, f );
    f.currentRawImage = raw;
    frames->enqueue( f );
    pthread_mutex_unlock( &frameLock );
    pthread_mutex_unlock( &roiLock );

    clock_gettime( CLOCK_REALTIME, &lanesStop );
    lanesThreadFrames++;
    lanesDeltaTimes += delta_t( &lanesStop, &lanesStart );
}

void LineDetector::findLeftLane( cv::Vec4i left, frame_s& f )
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
        ( 10 ) * ( CV_PI / 180 ),
        ( 65 ) * ( CV_PI / 180 )
    );

    while( !( foundLeft ) && i < lines.size() )
    {
        // sourced from OpenCV Hough tutorial:
        float rho = lines[ i ][ 0 ], theta = lines[ i ][ 1 ];
        if( abs( rho ) > 90 && abs( rho ) < 150 )
        {
            double a = cos( theta );
            double b = sin( theta );
            double x0 = a * rho;
            double y0 = b * rho;
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
            f.leftPt1 = cv::Point( std::round( ret.x ), std::round( ret.y ) ) + roiPoints[ 0 ];
        }
        else
        {
            foundLeft = false;
        }
        if( intersection( cv::Point( 0, roi.rows - 1 ), cv::Point( 1, roi.rows - 1 ), pt1, pt2, ret ) )
        {
            f.leftPt2 = cv::Point( round( ret.x ), round( ret.y ) ) + roiPoints[ 0 ];
        }
        else
        {
            foundLeft = false;
        }
    }
}

void LineDetector::findRightLane( cv::Vec4i right, frame_s& f )
{

    vector< cv::Vec3f > lines;
    uint32_t i = 0;
    cv::HoughLines(
        roi,            // image
        lines,          // lines
        1,              // rho resolution of accumulator in pixels
        CV_PI / 180,    // theta resolution of accumulator
        30,             // accumulator threshold, only lines >threshold returned
        0,              // srn - set to 0 for classical Hough
        0,              // stn - set to 0 for classical Hough
        2.007129,       // minimum theta
        2.967060        // maximum theta
    );


    while( !( foundRight ) && i < lines.size() )
    {
        float rho = lines[ i ][ 0 ], theta = lines[ i ][ 1 ];
        if( abs( rho ) > 150 && abs( rho ) < 300 )
        {
            double a = cos( theta );
            double b = sin( theta );
            double x0 = a * rho;
            double y0 = b * rho;
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
            f.rightPt1 = cv::Point( ret ) + roiPoints[ 0 ];
        }
        else
        {
            foundRight = false;
        }

        if( intersection( cv::Point( 0, roi.rows - 1 ), cv::Point( 1, roi.rows - 1 ), pt1, pt2, ret ) )
        {
            f.rightPt2 = cv::Point( ret ) + roiPoints[ 0 ];
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
        LogDebug( "Aborting carDetection" );
        carDetectionThread->shutdown();
        return;
    }
    sem_wait( semS3 );
    clock_gettime( CLOCK_REALTIME, &carsStart );

    cv::Mat raw;
    cv::Mat carImage;

    std::vector< cv::Rect > tmpVehicle;

    pthread_mutex_lock( &frameLock );
    if( frames->isEmpty() )
    {
        pthread_mutex_unlock( &frameLock );
        return;
    }
    frame_s f = frames->dequeue();
    if( f.currentRawImage.empty() )
    {
        frames->enqueue( f );
        pthread_mutex_unlock( &frameLock );
        return;
    }
    else
    {
        raw = f.currentRawImage;
        carImage = raw.clone();
        cv::Mat carImageHalf = carImage( cv::Rect( carPoints[ 0 ], carPoints[ 2 ] ) );
        cv::Mat gray;
        cv::cvtColor( carImageHalf, gray, cv::COLOR_BGR2GRAY );
        myClassifier.detectMultiScale( gray, tmpVehicle, 1.2, 4, 0, cv::Size( 16, 16 ), gray.size() );
    }

    if( tmpVehicle.size() == 0 )
    {
        // LogTrace( "No detected cars!" );
        // no cars detected in image
    }
    else if( f.vehicle.size() == 0 )
    {
        // LogTrace( "Adding detected cars!" );
        // raw image has been updated by LaneDetection but
        // we don't have any items in vehicle vector
        f.vehicle = tmpVehicle;
    }
    frames->enqueue( f );
    pthread_mutex_unlock( &frameLock );

    clock_gettime( CLOCK_REALTIME, &carsStop );
    carsThreadFrames++;
    carsDeltaTimes += delta_t( &carsStop, &carsStart );
    sem_post( semS4 );
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
    // myVideoWriter.write( myVehiclesImage );

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