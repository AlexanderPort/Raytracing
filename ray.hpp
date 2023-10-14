#ifndef RAY_HPP
#define RAY_HPP

#include "linalg.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct Ray {
    linalg::vector3 origin;
    linalg::vector3 direction;
    __host__ __device__ Ray(const linalg::vector3& origin, 
                            const linalg::vector3& direction) {
        this->origin = origin;
        this->direction = direction;
    }

    __host__ __device__ linalg::vector3 at(float t) {
        return origin + t * direction;
    }
} Ray;


#endif //RAY_HPP