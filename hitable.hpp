#ifndef HITABLE_HPP
#define HITABLE_HPP

#include "ray.hpp"
#include "aabb.hpp"
#include "linalg.hpp"

class material;

struct hit_record {
    float t;
    bool front_face;
    material *material;
    linalg::vector3 normal;
    linalg::vector3 position;
    linalg::vector3 intensity;


    __device__ void set_face_normal(
        const linalg::vector3& ro, const linalg::vector3& rd, 
        const linalg::vector3& outward_normal) {
        front_face = linalg::dot(rd, outward_normal) <= 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hitable  {
    public:
        virtual __device__ bool hit(const linalg::vector3& ro, const linalg::vector3& rd, 
                                    float tmin, float tmax, hit_record& record) const = 0;
        virtual __host__ __device__ bool bounding_box(aabb& box) const = 0;
};

#endif