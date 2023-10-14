#ifndef AABB_HPP
#define AABB_HPP

#include "linalg.hpp"

class aabb {
    public:
        __host__ __device__ aabb() {}
        __host__ __device__ aabb(const linalg::vector3& a, const linalg::vector3& b) { 
            this->minimum = a; this->maximum = b;
        }

        __device__ bool hit(const linalg::vector3& ro, 
                            const linalg::vector3& rd, 
                            float tmin, float tmax) const {
                     
            float t0; float d = 1.0f / rd.x;
            float t1 = (minimum.x - ro.x) * d;
            float t2 = (maximum.x - ro.x) * d;
            if (d < 0.0f) { t0 = t1; t1 = t2; t2 = t0; }
            tmin = t1 > tmin ? t1 : tmin;
            tmax = t2 < tmax ? t2 : tmax;
            if (tmax <= tmin) return false;

            d = 1.0f / rd.y;
            t1 = (minimum.y - ro.y) * d;
            t2 = (maximum.y - ro.y) * d;
            if (d < 0.0f) { t0 = t1; t1 = t2; t2 = t0; }
            tmin = t1 > tmin ? t1 : tmin;
            tmax = t2 < tmax ? t2 : tmax;
            if (tmax <= tmin) return false;
            
            d = 1.0f / rd.z;
            t1 = (minimum.z - ro.z) * d;
            t2 = (maximum.z - ro.z) * d;
            if (d < 0.0f) { t0 = t1; t1 = t2; t2 = t0; }
            tmin = t1 > tmin ? t1 : tmin;
            tmax = t2 < tmax ? t2 : tmax;
            if (tmax <= tmin) return false;
            
            return true;
        }

        static __host__ __device__ aabb surrounding_box(const aabb& box1, const aabb& box2) {
            linalg::vector3 small(fminf(box1.minimum.x, box2.minimum.x),
                                  fminf(box1.minimum.y, box2.minimum.y),
                                  fminf(box1.minimum.z, box2.minimum.z));
            linalg::vector3 big(fmaxf(box1.maximum.x, box2.maximum.x),
                                fmaxf(box1.maximum.y, box2.maximum.y),
                                fmaxf(box1.maximum.z, box2.maximum.z));
            return aabb(small, big);
        }

        linalg::vector3 minimum;
        linalg::vector3 maximum;
};

#endif