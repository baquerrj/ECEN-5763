/*!
 * @file RingBuffer.h
 * @author Roberto J Baquerizo (roba8460@colorado.edu)
 * @brief
 * @version 1.0
 * @date 2022-08-05
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __RING_BUFFER_H__
#define __RING_BUFFER_H__

#include <memory>
#include <unistd.h>

//! @brief Template Ring Buffer Class
template < class T >
class RingBuffer
{

    //! @brief Private data members
    private:
    std::unique_ptr< T[] > buffer;  //!< Actual buffer hold items of type T
    size_t head = 0;  //!< Pointer to top of buffer
    size_t tail = 0;  //!< Pointer to bottom of buffer
    size_t maxSize;  //!< Maximum number of items ring can hold
    T emptyItem;  //!< used to clear buffer
    std::string name;
    public:
    //! Create a new Ring_Buffer.
    RingBuffer< T >( size_t maxSize, std::string name ) :
        buffer( std::unique_ptr< T[] >( new T[ maxSize ] ) ), maxSize( maxSize ), name( name )
    {
        assert( maxSize > 1 && buffer != nullptr );
        pthread_mutex_init( &lock, NULL );
    }

    ~RingBuffer< T >()
    {
        pthread_mutex_destroy( &lock );
        buffer.reset();
    }

    //! @brief Add an item to this ring buffer
    //! @param item
    bool enqueue( T item );

    //! Remove an item from this ring buffer and return it.
    //! @return T
    T dequeue();

    //! Return the item at the front of this ring buffer.
    //! @return T
    T front();

    //! Check if buffer is empty
    //! @returns true if empty
    //! @returns false otherwise
    bool isEmpty();

    //! Check if buffer is full
    //! @returns true if full
    //! @returns false otherwise
    bool isFull();

    //! Get the size of the buffer
    //! @returns number of items currently in buffer
    size_t size();

    private:
    pthread_mutex_t lock;
};

#endif  // __RING_BUFFER_H__