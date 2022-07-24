/**
 * @file houghclines.cpp
 * @brief This program demonstrates line finding with the Hough transform
 */

#include "lanedetector.hpp"


using namespace std;

const String LineDetector::SOURCE_WINDOW_NAME = "Source";
const String LineDetector::DETECTED_LANES_IMAGE = "Detected Lanes (in red)";
const String LineDetector::DETECTED_VEHICLES_IMAGE = "Detected Vehicles";
const String LineDetector::CAR_CLASSIFIER = "cars.xml";

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

    if( vehicle.size() > 0 )
    {
        printf( "Detected %ld vehicles\n\r", vehicle.size() );
    }
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
    myVideoWriter.write(myVehiclesImage);
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
    myVideoCapture.read( mySource );
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
