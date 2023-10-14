#ifndef HITABLE_LIST_HPP
#define HITABLE_LIST_HPP

#include "cuda_runtime.h"
#include "hitable.hpp"
#include "light.hpp"


class hitable_list: public hitable  {
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(hitable** objects, int num_objects,
                                light** lights, int num_lights) {
            this->lights = lights;
            this->objects = objects;
            this->num_lights = num_lights;
            this->num_objects = num_objects;
        }

        __device__ virtual bool hit(const linalg::vector3& ro, const linalg::vector3& rd, 
                                    float tmin, float tmax, hit_record& record) const {
            bool hit = false;
            hit_record rrecord;
            float closest = tmax;
            for (int i = 0; i < num_objects; i++) {
                if (objects[i]->hit(ro, rd, tmin, closest, rrecord)) {
                    hit = true; closest = rrecord.t; record = rrecord;
                }
            }
            /*
            linalg::vector3 diffuse_intensity = linalg::vector3(0.0f, 0.0f, 0.0f);
            linalg::vector3 specular_intensity = linalg::vector3(0.0f, 0.0f, 0.0f);
            for (int i = 0; i < num_lights; i++) {
                light* l = lights[i];
                linalg::vector3 direction = linalg::normalize(l->position - record.position);
                float reflection_power = linalg::dot(reflect(direction, record.normal), rd);
                specular_intensity += l->intensity * pow(max(0.0f, reflection_power), 1.0f);
                diffuse_intensity += l->intensity * max(0.0f, linalg::dot(direction, record.normal));
            }
            record.intensity *= diffuse_intensity;
            record.intensity *= specular_intensity;
            */ 
             
            return hit;
        };

        __host__ __device__ bool bounding_box(aabb& box) const {
            if (num_objects == 0) return false;
            aabb temp_box;
            bool first_box = true;
            for (int i = 0; i < num_objects; i++) {
                hitable* object = objects[i];
                if (!object->bounding_box(temp_box)) return false;
                box = first_box ? temp_box : aabb::surrounding_box(box, temp_box);
                first_box = false;
            }
            return true;
        }

        light** lights;
        int num_lights = 0;
        hitable **objects;
        int num_objects = 0;
};


#endif