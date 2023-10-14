#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "hitable.hpp"

class sphere: public hitable  {
    public:
        __host__ __device__ sphere() {}
        __host__ __device__ sphere(const linalg::vector3& center, float radius, material* material) {
            this->center = center;
            this->radius = radius;
            this->material = material;
        };

        __device__ bool hit(const linalg::vector3& ro, const linalg::vector3& rd, 
                            float tmin, float tmax, hit_record& record) const {
            linalg::vector3 oc = ro - center;
            float b = dot(oc, rd);
            float c = dot(oc, oc) - radius * radius;
            float h = b * b - c;
            if (h < 0.0f) return false; 
            h = sqrt(h); float t = -b - h;
            if (t > tmax || t < tmin) return false;
            record.t = t;
            record.material = material;
            record.position = ro + t * rd;
            record.normal = normalize(record.position - center);
            return true;
        }

    __host__ __device__ bool bounding_box(aabb& box) const {
        box = aabb(
            center - linalg::vector3(radius, radius, radius),
            center + linalg::vector3(radius, radius, radius));
        return true;
    }

    float radius;
    material* material;
    linalg::vector3 center;

};


#endif