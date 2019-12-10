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

const char *err_to_str(cl_int error)
{
    switch (error) {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:
            return "CL_INVALID_PROPERTY";

        default:
            return "UNKNOWN ERROR";
    }
}

bool check_error(const char* description, cl_int err_code)
{
    if(err_code != CL_SUCCESS)
    {
        printf("OpenCL Error: %s: %s\n", description, err_to_str(err_code));
        return false;
    }

    return true;
}

bool check_program_build(cl_program program, cl_device_id device_id, cl_int err_code)
{
    if(err_code == CL_SUCCESS) return true;

    check_error("Building program", err_code);

    if(err_code == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
    }

    return false;
}

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
    check_error("Retrieving device IDs", status);

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
    check_program_build(state.program, state.device_id, status);

    // Create the OpenCL kernel
    state.kernel = clCreateKernel(state.program, "rasterize_tris", &status);
    check_error("Creating kernel", status);

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
    opencl_init(ocl_state);

    // rcRasterizeTriangles_OpenCL(rcContext* ctx, const float* verts, const int nv,
    //                       const int* tris, const unsigned char* areas, const int nt,
    //                       rcHeightfield& solid, const int flagMergeThr, opencl_state& ocl_state)

    static float verts[] = {0.0f, 0.0f, 0.0f};
    static int tris[] = {0, 1, 2};
    static unsigned char areas[] = {42};

    rcHeightfield dummyHF;
    dummyHF.bmin[0] = 1.0f;
    dummyHF.bmin[1] = 42.0f;
    dummyHF.bmin[2] = 33.0f;
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
    rcAssert(sizeof(float) == sizeof(cl_float)) 

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
    cl_mem verts_buf = clCreateBuffer(ocl_state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, verts_buf_size, (void*)verts, &errcode);
    check_error("Creating vertex buffer", errcode);
    cl_mem tris_buf = clCreateBuffer(ocl_state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tris_buf_size, (void*)tris, &errcode);
    check_error("Creating tris buffer", errcode);
    cl_mem areas_buf = clCreateBuffer(ocl_state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, areas_buf_size, (void*)areas, &errcode);
    check_error("Creating areas buffer", errcode);

    cl_float3 hf_bmin{ solid.bmin[0], solid.bmin[1], solid.bmin[2] };
    cl_float3 hf_bmax{ solid.bmax[0], solid.bmax[1], solid.bmax[2] };

    // cl_int clSetKernelArg(cl_kernel kernel,
    //  cl_uint arg_index,
    //  size_t arg_size,
    //  constvoid* arg_value);
    errcode  = clSetKernelArg(ocl_state.kernel, 0, sizeof(cl_mem), &verts_buf);
    errcode |= clSetKernelArg(ocl_state.kernel, 1, sizeof(cl_mem), &tris_buf);
    errcode |= clSetKernelArg(ocl_state.kernel, 2, sizeof(cl_mem), &areas_buf);
    errcode |= clSetKernelArg(ocl_state.kernel, 3, sizeof(cl_float3), &hf_bmin);
    errcode |= clSetKernelArg(ocl_state.kernel, 4, sizeof(cl_float3), &hf_bmax);
    errcode |= clSetKernelArg(ocl_state.kernel, 5, sizeof(float), &solid.cs);
    errcode |= clSetKernelArg(ocl_state.kernel, 6, sizeof(float), &solid.ch);
    errcode |= clSetKernelArg(ocl_state.kernel, 7, sizeof(int), &flagMergeThr);
    check_error("Passing kernel arguments", errcode);

    // Create a command queue
    // cl_command_queue clCreateCommandQueueWithProperties(
    //  cl_context context,    
    //  cl_device_id device,
    //  const cl_queue_properties* properties,    
    //  cl_int* errcode_ret);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ocl_state.context, ocl_state.device_id, 0, &errcode);
    check_error("Creating command queue", errcode);

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