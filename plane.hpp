#ifndef PLANE_HPP
#define PLANE_HPP

#include "hitable.hpp"


class xz_rect : public hitable {
    public:
        __device__ xz_rect() {}

        __device__ xz_rect(float x1, float z1, float x2, float z2, float k, material* material) {
            this->x1 = x1; this->z1 = z1; 
            this->x2 = x2; this->z2 = z2;
            this->k = k; this->material = material;
        }

        __device__ bool hit(const linalg::vector3& ro, const linalg::vector3& rd, 
                            float tmin, float tmax, hit_record& record) const {
            float t = (k - ro.y) / rd.y;
            if (t < tmin || t > tmax) return false;
            auto x = ro.x + t * rd.x; auto z = ro.z + t * rd.z;
            if (x < x1 || z < z1 || x > x2 || z > z2) return false;
            record.t = t; record.set_face_normal(ro, rd, {0, 1, 0});
            record.material = material; record.position = ro + rd * t;
            return true;
        };

        __host__ __device__ bool bounding_box(aabb& box) const {
             return true;
        }

    public:
        material* material;
        float x1, z1, x2, z2, k;
};


class xy_rect : public hitable {
    public:
        __device__ xy_rect() {}

        __device__ xy_rect(float x1, float y1, float x2, float y2, float k, material* material) {
            this->x1 = x1; this->y1 = y1; 
            this->x2 = x2; this->y2 = y2;
            this->k = k; this->material = material;
        }

        __device__ bool hit(const linalg::vector3& ro, const linalg::vector3& rd, 
                            float tmin, float tmax, hit_record& record) const {
            float t = (k - ro.z) / rd.z;
            if (t < tmin || t > tmax) return false;
            auto x = ro.x + t * rd.x; auto y = ro.y + t * rd.y;
            if (x < x1 || y < y1 || x > x2 || y > y2) return false;
            record.t = t; record.set_face_normal(ro, rd, {0, 0, -1});
            record.material = material; record.position = ro + rd * t;
            return true;
        };

        __host__ __device__ bool bounding_box(aabb& box) const {
             return true;
        }

    public:
        material* material;
        float x1, y1, x2, y2, k;
};


class yz_rect : public hitable {
    public:
        __device__ yz_rect() {}

        __device__ yz_rect(float y1, float z1, float y2, float z2, float k, material* material) {
            this->y1 = y1; this->z1 = z1; 
            this->y2 = y2; this->z2 = z2;
            this->k = k; this->material = material;
        }

        __device__ bool hit(const linalg::vector3& ro, const linalg::vector3& rd, 
                            float tmin, float tmax, hit_record& record) const {
            float t = (k - ro.x) / rd.x;
            if (t < tmin || t > tmax) return false;
            auto y = ro.y + t * rd.y; auto z = ro.z + t * rd.z;
            if (y < y1 || z < z1 || y > y2 || z > z2) return false;
            record.t = t; record.set_face_normal(ro, rd, {1, 0, 0});
            record.material = material; record.position = ro + rd * t;
            return true;
        };

        __host__ __device__ bool bounding_box(aabb& box) const {
             return true;
        }

    public:
        material* material;
        float y1, z1, y2, z2, k;
};

#endif //PLANE