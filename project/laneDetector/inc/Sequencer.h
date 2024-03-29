/*!
 * @file Sequencer.h
 * @author Roberto J Baquerizo (roba8460@colorado.edu)
 * @brief
 * @version 1.0
 * @date 2022-08-05
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __SEQUENCER_H__
#define __SEQUENCER_H__

#include <FrameBase.h>
#include <configuration.h>
#include <semaphore.h>

#include <memory>
#include <string>

class CyclicThread;

class Sequencer : public FrameBase
{
public:
   Sequencer();
   ~Sequencer();

   void sequenceServices();
   static void* execute( void* context );

   static const uint32_t SEQUENCER_FREQUENCY = 20;  // 20Hz

private:
   uint8_t captureFrequency;
};


inline Sequencer& getSequencer()
{
   static std::unique_ptr< Sequencer > singleton( new Sequencer() );
   return *singleton;
}

#endif  // __SEQUENCER_H__
