#include "RecastOpenCL.h"

#include <stdio.h>
#include <stdint.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "RecastAssert.h"

#define MAX_SOURCE_SIZE 0x100000

struct opencl_state
{
    cl_device_id device_id;
    cl_context context;
    cl_kernel kernel;
    cl_program program;
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
    state.program = clCreateProgramWithSource(state.context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &status);
 
    // Build the program
    status = clBuildProgram(state.program, 1, &state.device_id, NULL, NULL, NULL);
    if(status != CL_SUCCESS) return false;
 
    // Create the OpenCL kernel
    state.kernel = clCreateKernel(state.program, "test_kernel", &status);
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

    opencl_state ocl_state;
    if(opencl_init(ocl_state))
    {
        printf("Init works.\n");
    }
    else
    {
        printf("Init borked.\n");
    }

    // rcRasterizeTriangles_OpenCL(rcContext* ctx, const float* verts, const int nv,
    //                       const int* tris, const unsigned char* areas, const int nt,
    //                       rcHeightfield& solid, const int flagMergeThr, opencl_state& ocl_state)

    static float verts[] = {0.0f, 0.0f, 0.0f};
    static int tris[] = {0, 1, 2};
    static unsigned char areas[] = {42};

    rcHeightfield dummyHF;
    rcRasterizeTriangles_GPU(nullptr,   // rcContext* ctx
                                verts,   // const float* verts
                                sizeof(verts)/sizeof(verts[0]),         // const int nv
                                tris,   // const int* tris
                                areas,   // const unsigned char* areas
                                sizeof(areas)/sizeof(areas[0]),         // const int nt
                                dummyHF,   // rcHeightfield& solid
                                ocl_state);

    opencl_terminate(ocl_state);
}

// ===========================================================================

/// Rasterizes an indexed triangle mesh into the specified heightfield.
///  @ingroup recast
///  @param[in,out] ctx             The build context to use during the operation.
///  @param[in]     verts           The vertices. [(x, y, z) * @p nv]
///  @param[in]     nv              The number of vertices.
///  @param[in]     tris            The triangle indices. [(vertA, vertB, vertC) * @p nt]
///  @param[in]     areas           The area id's of the triangles. [Limit: <= #RC_WALKABLE_AREA] [Size: @p nt]
///  @param[in]     nt              The number of triangles.
///  @param[in,out] solid           An initialized heightfield.
///  @param[in]     flagMergeThr    The distance where the walkable flag is favored over the non-walkable flag. 
///                                 [Limit: >= 0] [Units: vx]
///  @returns True if the operation completed successfully.
bool rcRasterizeTriangles_GPU(rcContext* ctx, const float* verts, const int nv,
                          const int* tris, const unsigned char* areas, const int nt,
                          rcHeightfield& solid, opencl_state& ocl_state, const int flagMergeThr /* = 1 */)
{
    // Make sure whatever architecture that is, the sizes are consistent.
    // todo[drabard]: This could be a static assert.
    rcAssert(sizeof(int) == sizeof(cl_int)) 

    // Put things into memory
    // Constant:
    // solid.bmin, solid.bmax, solid.cs, solid.ch, 1.0f/solid.cs, 1.0f/solid.ch, flagMergeThr
    // Global:
    // verts [nv], tris [nt], areas [nt], 
    //      todo[drabard]: !!!output!!!

    cl_int errcode; 

    size_t verts_buf_size = nv * 3 * sizeof(float);
    size_t tris_buf_size = nt * 3 * sizeof(int);
    size_t areas_buf_size = nt * sizeof(unsigned char);
    // cl_mem clCreateBuffer(cl_context context,
    //     cl_mem_flags flags,
    //     size_t size,
    //     void* host_ptr,
    //     cl_int* errcode_ret);
    cl_mem verts_buf = clCreateBuffer(ocl_state.context, CL_MEM_READ_ONLY, verts_buf_size, 0, &errcode);
    errcode != CL_SUCCESS ? printf("Failed to create vertex buffer.\n") : printf("Verts buffer created.\n");
    cl_mem tris_buf = clCreateBuffer(ocl_state.context, CL_MEM_READ_ONLY, tris_buf_size, 0, &errcode);
    errcode != CL_SUCCESS ? printf("Failed to create tris buffer.\n") : printf("Tris buffer created.\n");
    cl_mem areas_buf = clCreateBuffer(ocl_state.context, CL_MEM_READ_ONLY, areas_buf_size, 0, &errcode);
    errcode != CL_SUCCESS ? printf("Failed to create areas buffer.\n") : printf("Areas buffer created.\n");

    // Create a command queue
    // cl_command_queue clCreateCommandQueueWithProperties(
    //  cl_context context,    
    //  cl_device_id device,
    //  const cl_queue_properties* properties,    
    //  cl_int* errcode_ret);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ocl_state.context, ocl_state.device_id, 0, &errcode);
    errcode != CL_SUCCESS ? printf("Failed to create command queue.\n") : printf("Created command queue no probs.\n");

    // Write the input buffers.
    // todo[drabard]: These should be non-blocking.
    // cl_int clEnqueueWriteBuffer(cl_command_queue command_queue,
    //     cl_mem buffer,
    //     cl_bool blocking_write,
    //     size_t offset,
    //     size_t size,
    //     constvoid* ptr,
    //     cl_uint num_events_in_wait_list,
    //     const cl_event* event_wait_list,
    //     cl_event* event);    
    errcode = clEnqueueWriteBuffer(queue, verts_buf, CL_TRUE, 0, verts_buf_size, verts, 0, 0, 0);
    errcode != CL_SUCCESS ? printf("Failed to write vertex buffer.\n") : printf("Vertex buffer write OK.\n");
    errcode = clEnqueueWriteBuffer(queue, tris_buf, CL_TRUE, 0, tris_buf_size, tris, 0, 0, 0);
    errcode != CL_SUCCESS ? printf("Failed to write tris buffer.\n") : printf("Tris buffer write OK.\n");
    errcode = clEnqueueWriteBuffer(queue, areas_buf, CL_TRUE, 0, areas_buf_size, areas, 0, 0, 0);
    errcode != CL_SUCCESS ? printf("Failed to write areas buffer.\n") : printf("Areas buffer write OK.\n");

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