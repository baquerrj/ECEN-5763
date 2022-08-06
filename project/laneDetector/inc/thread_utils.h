/*!
 * @file thread_utils.h
 * @author Roberto J Baquerizo (roba8460@colorado.edu)
 * @brief
 * @version 1.0
 * @date 2022-08-05
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __THREAD_UTILS_H__
#define __THREAD_UTILS_H__

#include <pthread.h>
#include <string>



extern int numCpus;
extern int cpuMain;
extern int cpuSequencer;
extern int cpuCapture;
extern int cpuLaneDetection;
extern int cpuSignDetection;
extern int cpuReceiver;
extern int cpuCarDetection;
extern int cpuAnnotation;

extern const uint16_t MAX_THREADNAME_LENGTH;

struct ProcessParams
{
   int cpuId;
   int policy;
   int priority;
   int nice;

   bool operator==( const ProcessParams &that ) const;
   bool operator!=( const ProcessParams &that ) const;
};

struct ThreadConfigData
{
   bool isValid;
   std::string threadName;
   ProcessParams processParams;

public:
   ThreadConfigData(){};
   ThreadConfigData( const bool isValid,
                     const std::string &threadName,
                     const ProcessParams &processParams ) :
       isValid( isValid ),
       threadName( threadName ),
       processParams( processParams )
   {
   }
   bool operator==( const ThreadConfigData &that ) const;
   bool operator!=( const ThreadConfigData &that ) const;
};

void set_thread_cpu_affinity( const pthread_t &threadId,
                              const int cpu );

void modify_thread( const pthread_t &threadId,
                    const ProcessParams &processParams );

void configure_thread_attributes( const std::string &threadName,
                                  const ProcessParams &processParams,
                                  pthread_attr_t &threadAttr );

void create_thread( const std::string threadName,
                    pthread_t &threadId,
                    void *( *start_routine )(void *),
                    void *args,
                    const ProcessParams &processParams );

void set_this_thread_cpu_affinity( const int cpu,
                                   const std::string threadName );

void join_thread( pthread_t &thread, bool &threadIsAlive );
void cancel_and_join_thread( pthread_t &thread, bool &threadIsAlive );
void kill_and_join_thread( pthread_t &thread, bool &threadIsAlive, int signal );

extern const ProcessParams DEFAULT_PROCESS_PARAMS;
extern const ProcessParams VOID_PROCESS_PARAMS;
extern const ProcessParams CAPUTRE_PROCESS_PARAMS;

extern const ThreadConfigData CAPTURE_THREAD_CONFIG;

#endif  // __THREAD_UTILS_H__
