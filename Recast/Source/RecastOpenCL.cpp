#include "RecastOpenCL.h"

#include <stdio.h>
#include <stdint.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE 0x100000

struct opencl_state
{
    cl_device_id device_id;
    cl_context context;
};

static bool opencl_init(struct opencl_state& state)
{
    state.device_id = NULL;   
    state.context = 0;

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int status = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    status = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, 
            &state.device_id, &ret_num_devices);
    if(status != CL_SUCCESS) return false;

    // Create an OpenCL context
    state.context = clCreateContext( NULL, 1, &state.device_id, NULL, NULL, &status);

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("kernels.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        return false;
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(state.context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &status);
 
    // Build the program
    status = clBuildProgram(program, 1, &state.device_id, NULL, NULL, NULL);
    if(status != CL_SUCCESS) return false;
 
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "test_kernel", &status);
    if(status != CL_SUCCESS) return false;

    return true;
}

static void opencl_terminate(struct opencl_state& state)
{
   clReleaseContext(state.context);
}

void opencl_test()
{
	printf("Start OpenCL test\n");

    opencl_state ctx;
    if(opencl_init(ctx))
    {
        printf("Init works.\n");
    }
    else
    {
        printf("Init borked.\n");
    }

    opencl_terminate(ctx);
}

// ===========================================================================

struct rcContext;
struct rcHeightfield;
bool rcRasterizeTriangles_OpenCL(rcContext* ctx, const float* verts, const int /*nv*/,
                          const int* tris, const unsigned char* areas, const int nt,
                          rcHeightfield& solid, const int flagMergeThr)
{
    // rcAssert(ctx);

    // rcScopedTimer timer(ctx, RC_TIMER_RASTERIZE_TRIANGLES);
    
    // const float ics = 1.0f/solid.cs;
    // const float ich = 1.0f/solid.ch;
    // // Rasterize triangles.
    // for (int i = 0; i < nt; ++i)
    // {
    //     const float* v0 = &verts[tris[i*3+0]*3];
    //     const float* v1 = &verts[tris[i*3+1]*3];
    //     const float* v2 = &verts[tris[i*3+2]*3];
    //     // Rasterize.
    //     if (!rasterizeTri(v0, v1, v2, areas[i], solid, solid.bmin, solid.bmax, solid.cs, ics, ich, flagMergeThr))
    //     {
    //         ctx->log(RC_LOG_ERROR, "rcRasterizeTriangles: Out of memory.");
    //         return false;
    //     }
    // }

    // return true;
    return false;
}