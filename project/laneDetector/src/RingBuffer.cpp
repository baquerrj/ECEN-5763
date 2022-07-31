#include <RingBuffer.h>
#include "opencv2/objdetect.hpp"

template < class T >
void RingBuffer< T >::enqueue( T item )
{
   // if buffer is full, throw an error
   if ( isFull() )
      throw std::runtime_error( "buffer is full" );

   // insert item at back of buffer
   buffer[ tail ] = item;

   // increment tail
   tail = ( tail + 1 ) % maxSize;
}

// Remove an item from this circular buffer and return it.
template < class T >
T RingBuffer< T >::dequeue()
{
   // if buffer is empty, throw an error
   if ( isEmpty() )
      throw std::runtime_error( "buffer is empty" );

   // get item at head
   T item = buffer[ head ];

   // set item at head to be empty
   buffer[ head ] = emptyItem;

   // move head foward
   head = ( head + 1 ) % maxSize;

   // return item
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
   return head == tail;
}

// Return true if this circular buffer is full, and false otherwise.
template < class T >
bool RingBuffer< T >::isFull()
{
   return tail == ( head - 1 ) % maxSize;
}

// Return the size of this circular buffer.
template < class T >
size_t RingBuffer< T >::size()
{
   if ( tail >= head )
      return tail - head;
   return maxSize - (head - tail);
}

template void RingBuffer< cv::Mat >::enqueue( cv::Mat item );
template cv::Mat RingBuffer< cv::Mat >::dequeue();
template cv::Mat RingBuffer< cv::Mat >::front();
template bool RingBuffer< cv::Mat >::isEmpty();
template bool RingBuffer< cv::Mat >::isFull();
template size_t RingBuffer< cv::Mat >::size();
