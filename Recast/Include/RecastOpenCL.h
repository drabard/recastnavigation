#ifndef RECAST_OPENCL_H
#define RECAST_OPENCL_H

#include "Recast.h"

struct opencl_state;
opencl_state* create_opencl_state();
void destroy_opencl_state(opencl_state** ocl_state);

bool rcRasterizeTriangles_GPU(rcContext* ctx, const float* verts, const int nv,
						  const int* tris, const unsigned char* areas, const int nt,
						  rcHeightfield& solid, opencl_state& ocl_state, const int flagMergeThr = 1);

#endif // RECAST_OPENCL_H