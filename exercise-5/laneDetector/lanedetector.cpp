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
        Vec4i l = linesP[i];
        line( myLanesImage, Point( l[0], l[1] ), Point( l[2], l[3] ), Scalar( 0, 0, 255 ), 3, LINE_AA );
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
        rectangle( myVehiclesImage, vehicle[i], CV_RGB( 255, 0, 0 ) );
    }
}

// static const char* keys = {
//     "{help          h|false|show help message}"
//     "{@input         |./videos/GP010639.MP4|path to test video to process}"
//     "{store          ||path to create processed output video}"
//     "{show           |false|show intermediate steps}"
// };

// void printHelp( CommandLineParser* p_parser )
// {
//     printf( "The program uses Hough line detection and Haar feature detection to identify and mark\n"
//             "road lanes and cars on a video stream input\n\r" );
//     p_parser->printMessage();
//     printf( "Press the ESC key to exit the program.\n\r" );
// }

// int main( int argc, char** argv )
// {
//     CommandLineParser parser( argc, argv, keys );
//     if( parser.get<bool>( "help" ) )
//     {
//         printHelp( &parser );
//         exit( 0 );
//     }

//     bool show = parser.get<bool>( "show" );
//     String store = parser.get<String>( "store" );
//     String videoInput = parser.get<String>( "@input" );

//     if( videoInput.empty() )
//     {
//         printf( "Missing path to test video!\n\r" );
//         printHelp( &parser );
//         exit( -1 );
//     }
//     else
//     {
//         printf( "Using %s as source\n\r", videoInput.c_str() );
//     }

//     LineDetector* detector = new LineDetector( 0, videoInput );

//     if( detector == NULL )
//     {
//         exit( -1 );
//     }

//     if( not detector->loadClassifier( LineDetector::CAR_CLASSIFIER ) )
//     {
//         delete detector;
//         exit( -1 );
//     }

//     char winInput;

//     int framesProcessed = 0;

//     while( true )
//     {
//         detector->readFrame();
//         if( detector->isFrameEmpty() )
//         {
//             break;
//         }
//         detector->prepareImage();
//         detector->detectLanes();
//         detector->detectCars();

//         framesProcessed++;

//         if( show )
//         {
//             detector->showLanesImage();
//             detector->showVehiclesImage();
//             detector->showSourceImage();
//         }
//         else
//         {
//             detector->showLanesImage();
//         }

//         winInput = waitKey( 2 );
//         if( winInput == 27 )
//         {
//             break;
//         }
//     }

//     if( detector != NULL )
//     {
//         delete detector;
//     }

//     return 0;
// }
