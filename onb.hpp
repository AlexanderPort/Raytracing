#ifndef ONB_HPP
#define ONB_HPP

#include "linalg.hpp"

class onb {
    public:
        __device__ onb() {}

        __device__ linalg::vector3 local(float a, float b, float c) const {
            return a * u + b * v + c * w;
        }

        __device__ linalg::vector3 local(const linalg::vector3& a) const {
            return a.x * u + a.y * v + a.z * w;
        }

        __device__ void build_from_w(const linalg::vector3& n) {
            w = linalg::normalize(n);
            linalg::vector3 a = (fabs(w.x) > 0.9f) ? linalg::vector3(0, 1, 0) : 
                                                     linalg::vector3(1, 0, 0);
            v = linalg::normalize(linalg::cross(w, a)); u = linalg::cross(w, v);
        };

    public:
        linalg::vector3 u, v, w;
};


#endif
