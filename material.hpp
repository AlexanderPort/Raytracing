#ifndef MATERIAL_HPP
#define MATERIAL_HPP

struct hit_record;

#include "hitable.hpp"

static __device__ linalg::vector3 reflect(const linalg::vector3& v, const linalg::vector3& n) {
    return v - 2 * linalg::dot(v, n) * n;
}

static __device__ bool refract(const linalg::vector3& v, const linalg::vector3& n, 
                               float ni_over_nt, linalg::vector3& refracted) {
    linalg::vector3 uv = linalg::normalize(v);
    float dt = linalg::dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    } else return false;
}

static __device__ linalg::vector3 random_in_unit_sphere(curandState *random) {
    linalg::vector3 rand = linalg::random_vector(random);
	float theta = rand.x * 2.0f * 3.14159265f;
	float phi = acos(2.0f * rand.y - 1.0);
	float r = pow(rand.z, 1.0f / 3.0f);
	float x = r * sin(phi) * cos(theta);
	float y = r * sin(phi) * sin(theta);
	float z = r * cos(phi);
	return {x, y, z};
}

class material {
    public:
        linalg::vector3 albedo;

        virtual __device__ linalg::vector3 emitted(const linalg::vector3& position) const {
            return linalg::vector3(0.0f, 0.0f, 0.0f);
        }

        virtual __device__ bool scatter(
            const linalg::vector3& iro, const linalg::vector3& ird, 
            const hit_record& record, linalg::vector3& attenuation,
            linalg::vector3& oro, linalg::vector3& ord, curandState* random
        ) const = 0;
};

class lambertian : public material {
    public:
        __host__ __device__ lambertian(const linalg::vector3& albedo) {
            this->albedo = albedo;
        }

        virtual __device__ bool scatter(
            const linalg::vector3& iro, const linalg::vector3& ird, 
            const hit_record& record, linalg::vector3& attenuation,
            linalg::vector3& oro, linalg::vector3& ord, curandState* random
        ) const override {
            ord = record.normal;
            //ord += random_in_unit_sphere(random);
            oro = record.position;
            attenuation = albedo;
            return true;
        }
};

class metal : public material {
    public:
        float alpha = 0.95f;
        __host__ __device__ metal(const linalg::vector3& albedo) {
            this->albedo = albedo;
        }
        __host__ __device__ metal(const linalg::vector3& albedo, float alpha) {
            this->albedo = albedo; this->alpha = alpha;
        }

        virtual __device__ bool scatter(
            const linalg::vector3& iro, const linalg::vector3& ird, 
            const hit_record& record, linalg::vector3& attenuation,
            linalg::vector3& oro, linalg::vector3& ord, curandState* random
        ) const override {
            ord = reflect(ird, record.normal);
            //ord += random_in_unit_sphere(random);
            oro = record.position; attenuation = albedo;

            linalg::vector3 r = 2 * linalg::random_vector(random) - 1;
            linalg::vector3 diffuse = r * dot(r, record.normal);
            ord = normalize((1.0f - alpha) * diffuse + alpha * ord);
            return (linalg::dot(ord, record.normal) >= 0);
        }
};


class dielectric : public material {
    public:
        __host__ __device__ dielectric(float refraction) : refraction(refraction) {}

        virtual __device__ bool scatter(
            const linalg::vector3& iro, const linalg::vector3& ird, 
            const hit_record& record, linalg::vector3& attenuation,
            linalg::vector3& oro, linalg::vector3& ord, curandState* random
        ) const override {
            linalg::vector3 outward_normal;

            linalg::vector3 reflected = reflect(ird, record.normal);
            float ni_over_nt; attenuation = {1.0f, 1.0f, 1.0f};

            linalg::vector3 refracted; float probability; 
            float cosine; float length = linalg::length(ird);
            
            if (dot(ird, record.normal) > 0) {
                outward_normal = -record.normal; ni_over_nt = refraction;
                cosine = refraction * linalg::dot(ird, record.normal) / length;
            } else {
                outward_normal = record.normal; ni_over_nt = 1.0 / refraction;
                cosine = -linalg::dot(ird, record.normal) / length;
            }

            if (refract(ird, outward_normal, ni_over_nt, refracted)) {
            probability = schlick(cosine, refraction);
            } else { probability = 1.0f; }

            if (curand_uniform(random) < probability) {
                     oro = record.position; ord = reflected;
            } else { oro = record.position; ord = refracted;}


            return true;
        }

    public:
        float refraction;

    private:
        static __device__ float schlick(float cosine, float refraction) {
            float r0 = (1 - refraction) / (1 + refraction);
            r0 = r0 * r0; return r0 + (1 - r0) * pow((1 - cosine), 5.0f);
        }
};


class diffuse_light : public material  {
    public:
        __host__ __device__ diffuse_light(const linalg::vector3& albedo) {
            this->albedo = albedo;
        }

        virtual __device__ bool scatter(
            const linalg::vector3& iro, const linalg::vector3& ird, 
            const hit_record& record, linalg::vector3& attenuation,
            linalg::vector3& oro, linalg::vector3& ord, curandState* random
        ) const override {
            attenuation = albedo;
            ord = record.normal;
            //ord = reflect(ird, record.normal);
            //ord = 2 * linalg::random_vector(random) - 1;
            oro = record.position;
            return true;
        }

        virtual __device__ linalg::vector3 emitted(const linalg::vector3& position) const override {
            return albedo;
        }
};

#endif
