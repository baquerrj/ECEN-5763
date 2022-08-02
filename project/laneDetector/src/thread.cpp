#include "Logger.h"
#include <pthread.h>
#include <thread.hpp>
#include "Logger.h"
ThreadBase::~ThreadBase()
{}

CyclicThread::CyclicThread( const ThreadConfigData& configData,
                            void* ( *execute_ )( void* context ),
                            void* owner_,
                            bool readyForThread ) :
    execute( execute_ ),
    owner( owner_ )
{
    threadData = configData;
    threadIsAlive = false;
    name = configData.threadName;

    if( readyForThread )
    {
        initiateThread();
    }
}

CyclicThread::CyclicThread( const ThreadConfigData& configData ) :
    CyclicThread( configData, NULL, NULL, false )
{}

CyclicThread::~CyclicThread()
{
    // LogDebug( "Entered" );
    terminate();
    // LogDebug( "Exiting" );
}

void CyclicThread::setFunctionAndOwner( void* ( *execute_ )( void* context ),
                                        void* owner_ )
{
    owner = owner_;
    execute = execute_;
}

void CyclicThread::initiateThread()
{
    LogDebug( "Entered" );
    threadIsAlive = true;
    try
    {
        create_thread( threadData.threadName,
                       thread,
                       CyclicThread::threadFunction,
                       this,
                       threadData.processParams );
    }
    catch( const std::string& e )
    {
        LogFatal( "Caught exception: %s.", e.c_str() );
        threadIsAlive = false;
    }
    catch( const std::exception& e )
    {
        LogFatal( "Caught exception: %s.", e.what() );
        threadIsAlive = false;
    }
    LogDebug( "Exiting" );
}

void CyclicThread::terminate()
{
    // LogDebug( "Entered" );
    cancel_and_join_thread( thread, threadIsAlive );
    // LogDebug( "Exiting" );
}

void* CyclicThread::cycle()
{
    while( threadIsAlive )
    {
        execute( owner );
    }
    LogInfo( "thread shutting down: %s", threadData.threadName.c_str() );
    pthread_exit( NULL );
    return NULL;
}

void* CyclicThread::threadFunction( void* context )
{
    return ( ( CyclicThread* )context )->cycle();
}
