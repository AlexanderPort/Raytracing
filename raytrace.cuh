#pragma once
#include <cuComplex.h>
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"

#include "plane.hpp"
#include "light.hpp"
#include "linalg.hpp"
#include "camera.hpp"
#include "sphere.hpp"
#include "hitable.hpp"
#include "material.hpp"
#include "bvh_node.hpp"
#include "hitable_list.hpp"

#include <stdio.h>

namespace raytracing {
    static int state = 0;
    static bool init = false;
    static light** cu_lights;
    static hitable** cu_world;
    static curandState* random;
    static hitable** cu_objects;
    static int cu_num_lights = 10;
    static int cu_num_objects = 100;
    static unsigned int* cu_screen_buffer = nullptr;
    static unsigned int* cu_environment_buffer = nullptr;
	void mandelbrot(unsigned int* screen_buffer, int w, int h, double dx, double dy,
                    double min_x, double max_x, double min_y, double max_y);
    void raytrace(unsigned int* screen_buffer, int sw, int sh, 
                  unsigned int* environment_buffer, int ew, int eh, 
                  float delta, const Camera& camera);
    void free();
}