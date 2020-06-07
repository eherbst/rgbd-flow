/*
 * edited by Evan from Louis' original H file in bl_filter to be a proper c++ module
 *
 * Evan Herbst
 * 6 / 21 / 10
 */

#ifndef PRIMESENSE_QUANTIZATION_V5_H
#define PRIMESENSE_QUANTIZATION_V5_H

#include <cassert>

#define DEPTH_BY_INDEX_ARRAY_SIZE 1136
extern const unsigned short depthByIndex[DEPTH_BY_INDEX_ARRAY_SIZE];

// the flipside of depthByIndex is indexByDepth
#define INDEX_BY_DEPTH_ARRAY_SIZE 9870 
extern const unsigned short indexByDepth[INDEX_BY_DEPTH_ARRAY_SIZE];

inline float getFloatDepthFromFloatIndex(float floatIndex) {
  assert (floatIndex >= 0.0);
  unsigned int index = (unsigned int) floatIndex;
  if ( index+1 < DEPTH_BY_INDEX_ARRAY_SIZE ) {
    // do interpolation
    return depthByIndex[index] + ( depthByIndex[index+1] - depthByIndex[index] ) * (floatIndex - index);
  } else {
    return depthByIndex[DEPTH_BY_INDEX_ARRAY_SIZE-1];
  }
}

inline unsigned short getDepthFromIndex(unsigned short index) {
  if ( index+1 < DEPTH_BY_INDEX_ARRAY_SIZE ) {
    return depthByIndex[index];
  } else {
    return depthByIndex[DEPTH_BY_INDEX_ARRAY_SIZE-1];
  }
}


inline unsigned short getIndexFromDepth(unsigned short depth) {
  if ( depth < INDEX_BY_DEPTH_ARRAY_SIZE ) {
    return indexByDepth[depth];
  } else {
    return indexByDepth[INDEX_BY_DEPTH_ARRAY_SIZE-1];
  }
}

#endif //header
