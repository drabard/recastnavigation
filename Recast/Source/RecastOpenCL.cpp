#include "RecastOpenCL.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "RecastAssert.h"
#include "RecastAlloc.h"

#define MAX_SOURCE_SIZE 0x100000

#include <sys/time.h>

int64_t getPerfTime_()
{
    timeval now;
    gettimeofday(&now, 0);
    return (int64_t)now.tv_sec*1000000L + (int64_t)now.tv_usec;
}

int getPerfTimeUsec_(const int64_t duration)
{
    return (int)duration;
}

struct scope_timer {
    scope_timer(const char* label, cl_command_queue queue)
    : label(label)
    , queue(queue)
    {
        duration = getPerfTime_();
    }

    ~scope_timer()
    {
        clFinish(queue);
        duration = getPerfTime_() - duration;
        printf("%s: %dus\n", label, duration);
    }

    const char* label;
    int64_t duration;
    cl_command_queue queue;
};

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
   clReleaseProgram(state.program);
   clReleaseKernel(state.kernel);
}

opencl_state* create_opencl_state()
{
    opencl_state* res = (opencl_state*)rcAlloc(sizeof(opencl_state), RC_ALLOC_PERM);
    if(res == NULL) return res;
    opencl_init(*res);
    return res;
}

void destroy_opencl_state(opencl_state** ocl_state)
{
    opencl_terminate(**(ocl_state));
    *ocl_state = NULL;
}

void opencl_test()
{
	printf("Start OpenCL test\n");

    opencl_state ocl_state;
    opencl_init(ocl_state);

    // rcRasterizeTriangles_OpenCL(rcContext* ctx, const float* verts, const int nv,
    //                       const int* tris, const unsigned char* areas, const int nt,
    //                       rcHeightfield& solid, const int flagMergeThr, opencl_state& ocl_state)

    static float verts[] = {0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 2.0f, 2.0f, -2.0f};
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
    // Make sure that the sizes are consistent across the host and device.
    // todo[drabard]: This could be a static assert.
    rcAssert(sizeof(int) == sizeof(cl_int));
    rcAssert(sizeof(float) == sizeof(cl_float));
    rcAssert(sizeof(char) == sizeof(cl_char));
    rcAssert(sizeof(unsigned short) == sizeof(cl_ushort));

    rcScopedTimer timer(ctx, RC_TIMER_RASTERIZE_TRIANGLES);

    cl_int errcode; 

    size_t verts_buf_size = nv * 3 * sizeof(float);
    size_t tris_buf_size = nt * 3 * sizeof(int);
    size_t areas_buf_size = nt * sizeof(unsigned char);

    size_t max_spans_per_tri = 1024;
    size_t max_spans = max_spans_per_tri * nt;
    size_t out_xy_buf_size = sizeof(cl_int)*2*max_spans;
    size_t out_sminmax_buf_size = sizeof(cl_ushort)*2*max_spans;

    // Create a command queue
    // cl_command_queue clCreateCommandQueueWithProperties(
    //  cl_context context,    
    //  cl_device_id device,
    //  const cl_queue_properties* properties,    
    //  cl_int* errcode_ret);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ocl_state.context, ocl_state.device_id, 0, &errcode);
    check_error("Creating command queue", errcode);

    cl_mem verts_buf;
    cl_mem tris_buf;
    cl_mem out_xy;
    cl_mem out_sminmax;
    cl_int* spans_xy;
    cl_ushort* spans_sminmax;
    {
        scope_timer t("Creating buffers", queue);

        // cl_mem clCreateBuffer(cl_context context,
        //     cl_mem_flags flags,
        //     size_t size,
        //     void* host_ptr,
        //     cl_int* errcode_ret);
        // Create input buffers.
        verts_buf = clCreateBuffer(ocl_state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, verts_buf_size, (void*)verts, &errcode);
        check_error("Creating vertex buffer", errcode);
        tris_buf = clCreateBuffer(ocl_state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tris_buf_size, (void*)tris, &errcode);
        check_error("Creating tris buffer", errcode);

        spans_xy = (cl_int*)rcAlloc(out_xy_buf_size, RC_ALLOC_TEMP);
        if(spans_xy == NULL) return false;
        spans_sminmax = (cl_ushort*)rcAlloc(out_sminmax_buf_size, RC_ALLOC_TEMP);
        if(spans_sminmax == NULL) return false;

        // Create output buffers.
        out_xy = clCreateBuffer(ocl_state.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, out_xy_buf_size, 0, &errcode);
        check_error("Creating out xy buffer", errcode);
        out_sminmax = clCreateBuffer(ocl_state.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, out_sminmax_buf_size, 0, &errcode);
        check_error("Creating out smin smax buffer", errcode);
    }

    {
        scope_timer t("Passing kernel arguments", queue);

        cl_float3 hf_bmin{ solid.bmin[0], solid.bmin[1], solid.bmin[2] };
        cl_float3 hf_bmax{ solid.bmax[0], solid.bmax[1], solid.bmax[2] };

        // cl_int clSetKernelArg(cl_kernel kernel,
        //  cl_uint arg_index,
        //  size_t arg_size,
        //  constvoid* arg_value);
        errcode  = clSetKernelArg(ocl_state.kernel, 0, sizeof(cl_mem), &verts_buf);
        errcode |= clSetKernelArg(ocl_state.kernel, 1, sizeof(cl_mem), &tris_buf);
        errcode |= clSetKernelArg(ocl_state.kernel, 2, sizeof(cl_mem), &out_xy);
        errcode |= clSetKernelArg(ocl_state.kernel, 3, sizeof(cl_mem), &out_sminmax);
        errcode |= clSetKernelArg(ocl_state.kernel, 4, sizeof(cl_float3), &hf_bmin);
        errcode |= clSetKernelArg(ocl_state.kernel, 5, sizeof(cl_float3), &hf_bmax);
        errcode |= clSetKernelArg(ocl_state.kernel, 6, sizeof(cl_float), &solid.cs);
        errcode |= clSetKernelArg(ocl_state.kernel, 7, sizeof(cl_float), &solid.ch);
        errcode |= clSetKernelArg(ocl_state.kernel, 8, sizeof(cl_int), &solid.width);
        errcode |= clSetKernelArg(ocl_state.kernel, 9, sizeof(cl_int), &solid.height);
        errcode |= clSetKernelArg(ocl_state.kernel, 10, sizeof(cl_int), &RC_SPAN_MAX_HEIGHT);
        errcode |= clSetKernelArg(ocl_state.kernel, 11, sizeof(cl_int), &max_spans_per_tri);
        check_error("Passing kernel arguments", errcode);
    }

    {
        scope_timer t("Running the kernel", queue);

        // cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
        //  cl_kernel kernel,
        //  cl_uint work_dim,
        //  const size_t* global_work_offset,
        //  const size_t* global_work_size,
        //  const size_t* local_work_size,
        //  cl_uint num_events_in_wait_list,
        //  const cl_event* event_wait_list,
        //  cl_event* event);
        const size_t global_work_size[] = {(size_t)nt};
        const size_t local_work_size[] = {(size_t)64};
        errcode = clEnqueueNDRangeKernel(queue, ocl_state.kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        check_error("Running the kernel", errcode);
    }

    {
        scope_timer t("Out buffer allocation", queue);


        errcode = clFinish(queue);
        check_error("Waiting for kernel to finish", errcode);
    }

    {
        scope_timer t("Reading out buffers", queue);

        errcode = clEnqueueReadBuffer( queue, out_xy, CL_FALSE, 0, out_xy_buf_size, spans_xy, 0, NULL, NULL );  
        check_error("Reading out_xy output buffer.", errcode);
        errcode = clEnqueueReadBuffer( queue, out_sminmax, CL_FALSE, 0, out_sminmax_buf_size, spans_sminmax, 0, NULL, NULL );  
        check_error("Reading out_sminmax output buffer.", errcode);
    }

    bool is_ok = true;
    {
        scope_timer t("Adding spans", queue);
        is_ok = true;
        int total_spans = 0;
        for(int ti = 0; ti < nt; ++ti)
        {
            char area = areas[ti];
            size_t toffset = ti*max_spans_per_tri*2;
            int processed_spans = 0;
            for(;;)
            {
                if(processed_spans == max_spans_per_tri) break;

                int idx = toffset + processed_spans * 2;
                int x = spans_xy[idx];
                if(x == -1) break; // -1 is a termination value.
                int y = spans_xy[idx + 1];
                assert(y >= 0);
                unsigned short smin = spans_sminmax[idx];
                unsigned short smax = spans_sminmax[idx + 1];
                if(!rcAddSpan(ctx, solid, x, y, smin, smax, area, flagMergeThr))
                {
                    is_ok = false;
                    break;
                }
                ++processed_spans;
            }

            total_spans += processed_spans;
        }

        printf("Generated %d spans\n", total_spans);
    }

    { 
        scope_timer t("Cleanup", 0);

        rcFree(spans_xy);
        rcFree(spans_sminmax);
        clReleaseMemObject(out_xy);
        clReleaseMemObject(out_sminmax);
        clReleaseMemObject(verts_buf);
        clReleaseMemObject(tris_buf);
        clReleaseCommandQueue(queue);
    }
 
    return is_ok;
}