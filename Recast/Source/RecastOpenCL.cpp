#include "RecastOpenCL.h"

#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

struct opencl_context
{
    cl_device_id device_id;
    cl_context context;
};

static bool opencl_init(struct opencl_context& context)
{
    context.device_id = NULL;   
    context.context = 0;

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int status = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if(status != CL_SUCCESS) goto error;

    status = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, 
            &context.device_id, &ret_num_devices);
    if(status != CL_SUCCESS) goto error;

    // Create an OpenCL context
    context.context = clCreateContext( NULL, 1, &context.device_id, NULL, NULL, &status);
    if(status != CL_SUCCESS) goto error;

    return true;

error:
    clReleaseContext(context.context);
    return false;
}

static void opencl_build_program()
{

}

static void opencl_terminate(struct opencl_context& context)
{
   clReleaseContext(context.context);
}

void opencl_test()
{
	printf("Start OpenCL test\n");

    opencl_context ctx;
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