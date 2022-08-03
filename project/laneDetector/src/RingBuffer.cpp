#include <RingBuffer.h>
#include "opencv2/objdetect.hpp"
#include "lanedetector.hpp"

template < class T >
void RingBuffer< T >::enqueue( T item )
{
    pthread_mutex_lock( &lock );
    // if buffer is full, throw an error
    if( isFull() )
    {
        pthread_mutex_unlock( &lock );
        throw std::runtime_error( "buffer is full" );
    }
    // insert item at back of buffer
    buffer[ tail ] = item;

    // increment tail
    tail = ( tail + 1 ) % maxSize;
    pthread_mutex_unlock( &lock );
}

// Remove an item from this circular buffer and return it.
template < class T >
T RingBuffer< T >::dequeue()
{
    pthread_mutex_lock( &lock );
    // if buffer is empty, throw an error
    if( isEmpty() )
    {
        pthread_mutex_unlock( &lock );
        throw std::runtime_error( "buffer is empty" );
    }

    // get item at head
    T item = buffer[ head ];

    // set item at head to be empty
    buffer[ head ] = emptyItem;

    // move head foward
    head = ( head + 1 ) % maxSize;

    // return item
    pthread_mutex_unlock( &lock );
    return item;
}

// Return the item at the front of this circular buffer.
template < class T >
T RingBuffer< T >::front()
{
    return buffer[ head ];
}

// Return true if this circular buffer is empty, and false otherwise.
template < class T >
bool RingBuffer<T >::isEmpty()
{
    bool empty = (head == tail);
    return empty;
}

// Return true if this circular buffer is full, and false otherwise.
template < class T >
bool RingBuffer< T >::isFull()
{
    bool full = ( tail == ( head - 1 ) % maxSize );
    return full;
}

// Return the size of this circular buffer.
template < class T >
size_t RingBuffer< T >::size()
{
    pthread_mutex_lock( &lock );
    size_t _size = 0;
    if( tail >= head )
    {
        _size = tail - head;
    }
    else
    {
        _size = maxSize - ( head - tail );
    }
    pthread_mutex_unlock( &lock );
    return _size;
}

template void RingBuffer< cv::Mat >::enqueue( cv::Mat item );
template cv::Mat RingBuffer< cv::Mat >::dequeue();
template cv::Mat RingBuffer< cv::Mat >::front();
template bool RingBuffer< cv::Mat >::isEmpty();
template bool RingBuffer< cv::Mat >::isFull();
template size_t RingBuffer< cv::Mat >::size();

template void RingBuffer< cv::Point >::enqueue( cv::Point item );
template cv::Point RingBuffer< cv::Point >::dequeue();
template cv::Point RingBuffer< cv::Point >::front();
template bool RingBuffer< cv::Point >::isEmpty();
template bool RingBuffer< cv::Point >::isFull();
template size_t RingBuffer< cv::Point >::size();

template void RingBuffer< std::vector< cv::Rect> >::enqueue( std::vector< cv::Rect> item );
template std::vector< cv::Rect> RingBuffer< std::vector< cv::Rect> >::dequeue();
template std::vector< cv::Rect> RingBuffer< std::vector< cv::Rect> >::front();
template bool RingBuffer< std::vector< cv::Rect> >::isEmpty();
template bool RingBuffer< std::vector< cv::Rect> >::isFull();
template size_t RingBuffer< std::vector< cv::Rect> >::size();

template void RingBuffer< LineDetector::frame_s >::enqueue( LineDetector::frame_s item );
template LineDetector::frame_s RingBuffer< LineDetector::frame_s >::dequeue();
template LineDetector::frame_s RingBuffer< LineDetector::frame_s >::front();
template bool RingBuffer< LineDetector::frame_s >::isEmpty();
template bool RingBuffer< LineDetector::frame_s >::isFull();
template size_t RingBuffer< LineDetector::frame_s >::size();
