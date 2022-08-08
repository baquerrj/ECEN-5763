/*!
 * @file thread.h
 * @author Roberto J Baquerizo (roba8460@colorado.edu)
 * @brief
 * @version 1.0
 * @date 2022-08-05
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __THREAD_H__
#define __THREAD_H__

#include <thread_utils.h>
#include "Logger.h"
class ThreadBase
{
    public:
    virtual ~ThreadBase() = 0;

    virtual void setFunctionAndOwner( void* ( *execute_ )( void* context ),
                                      void* owner_ ) = 0;
    virtual void initiateThread() = 0;
    virtual void terminate() = 0;
    virtual const char* getName() = 0;

    protected:
    ThreadConfigData threadData;
    pthread_t thread;
    bool threadIsAlive;
    void* ( *execute )( void* context );
    void* owner;
    std::string name;
};

class CyclicThread : public ThreadBase
{
    public:
    CyclicThread( const ThreadConfigData& configData,
                  void* ( *execute_ )( void* context ),
                  void* owner_,
                  bool readyForThread = true );
    CyclicThread( const ThreadConfigData& configData );
    virtual ~CyclicThread();

    virtual void setFunctionAndOwner( void* ( *execute_ )( void* context ),
                                      void* owner_ );
    virtual void initiateThread();
    virtual void terminate();
    virtual void shutdown();
    pthread_t getThreadId();
    bool isThreadAlive();
    inline const char* getName()
    {
        return name.c_str();
    }

    protected:
    void* ( *execute )( void* context );
    virtual void* cycle();
    static void* threadFunction( void* context );

    void* owner;
};

inline pthread_t CyclicThread::getThreadId()
{
    return thread;
}

inline bool CyclicThread::isThreadAlive()
{
    return threadIsAlive;
}

inline void CyclicThread::shutdown()
{
    if( threadIsAlive )
    {
        threadIsAlive = false;
    }
}
#endif  // __THREAD_H__