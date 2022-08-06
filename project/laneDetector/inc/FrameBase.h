/*!
 * @file FrameBase.h
 * @author Roberto J Baquerizo (roba8460@colorado.edu)
 * @brief
 * @version 1.0
 * @date 2022-08-04
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __FRAME_BASE_H__
#define __FRAME_BASE_H__

#include <semaphore.h>
#include <string.h>

#include <memory>

class CyclicThread;
struct ThreadConfigData;
class FrameBase
{
    public:
    /*!
     * @brief Construct a new Frame Base object
     *
     * @param config
     */
    FrameBase( const ThreadConfigData config );
    /*!
     * @brief Destroy the Frame Base object
     *
     */
    virtual ~FrameBase();

    /*!
     * @brief Shutdown threads
     *
     */
    virtual void shutdown();
    /*!
     * @brief Checks if object is alive
     *
     * @return true
     * @return false
     */
    virtual bool isAlive();
    /*!
     * @brief Checks if thread is alive
     *
     * @return true
     * @return false
     */
    virtual bool isThreadAlive();
    /*!
     * @brief Get the Thread Id of the cyclic thread
     *
     * @return pthread_t
     */
    virtual pthread_t getThreadId();
    /*!
     * @brief Perform jitter analysis
     *
     */
    virtual void jitterAnalysis();
    /*!
     * @brief Get the current frame count
     *
     * @return uint32_t
     */
    virtual uint32_t getFrameCount();
    /*!
     * @brief Set deadline for this object
     *
     * @param deadlineTime
     */
    virtual void setDeadline( double deadlineTime );
    protected:
    std::string name;   //!< String identifier
    double wcet;        //!< Worst-case execution time
    double aet;         //!< Average execution time
    double deadline;    //!< Processing deadline
    unsigned long long count;   //!< Number indicating number of processed frames
    uint32_t frameCount;    //!< Number indicating number of processed frames
    struct timespec start;  //!< To measure start time of the service
    struct timespec end;    //!< To measure end time of the service

    double* executionTimes;  //!< To store execution time for each iteration
    double* startTimes;      //!< To store start time for each iteration
    double* endTimes;        //!< To store end time for each iteration

    bool alive;             //!< True if Frame object is alive
    CyclicThread* thread;   //!< Thread used for processing
    uint32_t requiredIterations;    //!< Number of iterations before exiting
};


#endif  //__FRAME_BASE_H__
