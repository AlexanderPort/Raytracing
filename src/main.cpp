
#include <iostream>
#include "linalg/Matrix4x4.h"
#include "linalg/Quaternion.h"
#include <vector>
#include <chrono>
#include "rendering/Renderer.h"
#include "rendering/Texture.h"

#include "primitives/Object3D.h"
#include "primitives/Instance3D.h"

#include <SDL.h>
#include <SDL_image.h>

#define ITERATIONS 100



int belong(Quaternion& q, float c, int x, int y, int z) {
    int iterations = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        q = q + c; q = q * q;
        iterations += 1;
        if (-x > q.x || q.x > x ||
            -y > q.y || q.y > y ||
            -z > q.z || q.z > z) {
            return iterations;
        }
    }
    return iterations;
}


float random() {
    return rand() % 1001 / 1000.0f - 0.5f;
}





int main(int argc, char* argv[]) {
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    if (SDL_Init(SDL_INIT_VIDEO) == 0) {
        ////////////////////////////////////////////////////////////////////////////////

        SDL_Window* window = SDL_CreateWindow(
                "window",  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1000, 500,
                SDL_WINDOW_VULKAN | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_FULLSCREEN_DESKTOP);

        SDL_Surface* window_surface = SDL_GetWindowSurface(window);
        SDL_Renderer* renderer2D = SDL_CreateRenderer(
                window, -1, SDL_RENDERER_ACCELERATED);
        SDL_bool done = SDL_FALSE;


        ////////////////////////////////////////////////////////////////////////////////

        std::string assets_path = "C:/Projects/3D/assets/";
        std::string models_path = assets_path + "models/";
        std::string textures_path = assets_path + "textures/";

        Renderer renderer3D = Renderer(window_surface, renderer2D);
        Object3D cube_object = Object3D::load(models_path + "oxen.obj");
        Texture space_texture = Texture(textures_path + "oxen.jpg", renderer3D.format);
        cube_object.texture = &space_texture;
        float background_color = 0;
        std::vector<Instance3D> instances = std::vector<Instance3D>();
        int count = 1;
        instances.reserve(count);
        for (int i = 0; i < count; i++) {
            Instance3D instance = Instance3D(&cube_object);
            instance.scale = Vector4(1, 1, 1, 1);
            //instance.speed = Vector4(random() * 10, random() * 10, random() * 10, 0);
            instance.position.x = 0;
            instance.position.y = 0;
            instance.position.z = 0;
            instances.push_back(instance);
        }
        ////////////////////////////////////////////////////////////////////////////////
        while (!done) {
            SDL_Event event;
            SDL_FillRect(window_surface, nullptr, 0x000000);
            renderer3D.update();
            //particleSystem.update();
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    done = SDL_TRUE;
                }
                renderer3D.camera.control(event);
            }
            renderer3D.light = renderer3D.camera.position;
        //#pragma omp parallel forcube
            for (Instance3D& instance : instances) {
                instance.update();
                instance.render(&renderer3D);
                //instance.rotate_around(renderer3D.camera.position, {0.1f, 0, 0});
                //instance.rotation.y += 0.1f;
            }
            //particleSystem.draw();
            //std::cout << renderer3D.rasterizer.triangles << std::endl;
            //renderer3D.rasterizer.triangles = 0;
            
            SDL_UpdateWindowSurface(window);
/*
#pragma omp parallel for
            for (int z = 0; z < sizeZ; ++z) {
                float Z = 4 * z / sizeZ - 1;
                for (int y = 0; y < sizeY; ++y) {
                    float Y = 4 * y / sizeY - 1;
                    for (int x = 0; x < sizeX; ++x) {
                        float X = 4 * x / sizeX - 1;
                        Vector4 v = Vector4(x, y, z, 1);
                        Quaternion q = Quaternion(X, Y, Z, 0);
                        int iterations = belong(q, std::sin(c), 100, 100, 100);
                        //std::cout << iterations << std::endl;
                        int color = 0xFFFFFF * iterations / (float)ITERATIONS;
                        v = renderer3D.mViewProjection * v;
                        v.x /= v.w; v.y /= v.w;
                        if (0 < v.x && v.x < renderer3D.width && 0 < v.y && v.y < renderer3D.height) {
                            pixel_buffer[int(v.x + v.y * renderer3D.width)] = color;

                        }
                        //std::cout << int(v.x + v.y * renderer3D.width) << std::endl;
                    }
                }
            }
            SDL_UpdateWindowSurface(window);
            c -= 0.001;

        }
        */
        
    }
    if (renderer2D) {
        SDL_DestroyRenderer(renderer2D);
    }
    if (window) {
        SDL_DestroyWindow(window);
    }
    SDL_Quit();
    return 0;
}
}


/*
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cstring>

#ifdef __APPLE__
# include <OpenCL/opencl.h>      // для компьютеров на MacOsX
#else
# include <CL/cl.h>              // для компьютеров на Win\Linux указывайте путь к файлу cl.h
#endif

#define MAX_SRC_SIZE (0x100000)  // максимальный размер исходного кода кернеля


class CLLib
{
private:
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    std::string source;
    int *A, *B, *C;

    cl_mem memobjInput = NULL;
    cl_mem memobjOutput = NULL;

    cl_kernel kernel = NULL;

    size_t workGroups;
    size_t workItems;
    size_t dimSize;
    int size = 1024;


public:

    size_t inputSize;
    size_t outputSize;
    void* bufferIn;
    void* bufferOut;

    CLLib(std::string filename, std::string kernelName)
    {

        ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);


        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
        std::string line, lines;
        std::ifstream in("/home/alexander/Projects/CLionProjects/Graphics3D/src/vector.cl"); // окрываем файл для чтения
        if (in.is_open())
        {
            while (getline(in, line))
            {
                source += line;
            }
        }
        in.close();

        size_t source_size = source.length() + 1;
        const char* source_str = source.c_str();
        //strcpy_s(source_str, source_size * sizeof(char), source.c_str());

        program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                            (const size_t *)&source_size, &ret);


        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        A = new int [size]; B = new int [size]; C = new int [size];

        if (ret == CL_BUILD_PROGRAM_FAILURE) {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            // Allocate memory for the log
            char *log = (char *) malloc(log_size);

            // Get the log
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

            // Print the log
            printf("%s\n", log);
        }



        kernel = clCreateKernel(program, kernelName.c_str(), &ret);

        delete[] source_str;

    }


    void reinitDataContainers(size_t inputSize, size_t outputSize)
    {
        this->inputSize = inputSize;
        this->outputSize = outputSize;

        if(bufferIn){
            free(bufferIn);
        }
        if(bufferOut){
            free(bufferOut);
        }

        bufferIn = malloc(inputSize);
        bufferOut = malloc(outputSize);

        if(memobjInput){
            ret = clReleaseMemObject(memobjInput);
        }
        if(memobjOutput){
            ret = clReleaseMemObject(memobjOutput);
        }


        memobjInput = clCreateBuffer(context, CL_MEM_READ_WRITE, inputSize, 0, &ret);
        memobjOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputSize, 0, &ret);

    }
    void build(size_t dimSize, size_t workGroups, size_t workItems)
    {
        this->workGroups = workGroups;
        this->workItems = workItems;
        this->dimSize = dimSize;

        clEnqueueWriteBuffer(command_queue, memobjInput, CL_TRUE, 0, inputSize, bufferIn, 0, NULL, NULL);

        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjInput);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjOutput);

    }

    void execute()
    {
        ret = clEnqueueNDRangeKernel(command_queue, kernel, dimSize, 0, &workGroups, &workItems, 0, NULL, NULL);

        clEnqueueReadBuffer(command_queue, memobjOutput, CL_TRUE, 0, outputSize, bufferOut, 0, NULL, NULL);
        //println("delta: "+ toString(Timer::getTimeNanoSeconds() - curTime));
    }

    void release()
    {

        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(memobjInput);
        ret = clReleaseMemObject(memobjOutput);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);

        free(bufferIn);
        free(bufferOut);
    }
};

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>

#define INP_SIZE (1024)

// Simple compute kernel which computes the cube of an input array

const char *KernelSource = "\n" \
"__kernel void square( __global float* input, __global float* output, \n" \
" const unsigned int count) {            \n" \
" int i = get_global_id(0);              \n" \
" if(i < count) \n" \
" output[i] = input[i] * input[i] * input[i]; \n" \
"}                     \n" ;

int main(int argc, char** argv)
{

    int err; // error code
    float data[INP_SIZE]; // original input data set to device
    float results[INP_SIZE]; // results returned from device
    unsigned int correct; // number of correct results returned

    size_t global; // global domain size
    size_t local; // local domain size

    cl_device_id device_id; // compute device id
    cl_context context; // compute context
    cl_command_queue commands; // compute command queue
    cl_program program; // compute program
    cl_kernel kernel; // compute kernel
    cl_mem input; // device memory used for the input array
    cl_mem output; // device memory used for the output array

    // Fill our data set with random values
    int i = 0;
    unsigned int count = INP_SIZE;

    for(i = 0; i < count; i++)
        data[i] = rand() / 50.00;


    // Connect to a compute device
    // If want to run your kernel on CPU then replace the parameter CL_DEVICE_TYPE_GPU
    // with CL_DEVICE_TYPE_CPU

    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }


    // Create a compute context
    //Contexts are responsible for managing objects such as command-queues, memory, program and kernel objects and for executing kernels on one or more devices specified in the context.

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);

    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    clFinish(commands);

    // Read back the results from the device to verify the output
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    // Print obtained results from OpenCL kernel
    for(i=0; i<count; i++ )
    {
        printf("result[%d] = %f", i, results[i]) ;
    }

    // Cleaning up
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}
*/