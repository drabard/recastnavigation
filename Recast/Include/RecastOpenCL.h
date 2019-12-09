#ifndef RECAST_OPENCL_H
#define RECAST_OPENCL_H

#include "Recast.h"

bool rcRasterizeTriangles_GPU(rcContext* ctx, const float* verts, const int nv,
						  const int* tris, const unsigned char* areas, const int nt,
						  rcHeightfield& solid, const int flagMergeThr = 1);

void opencl_test();

#endif // RECAST_OPENCL_H