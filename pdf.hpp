#ifndef PDF_HPP
#define PDF_HPP

#include "linalg.hpp"


#ifndef M_PI
#define M_PI 3.1415f
#endif


#include "onb.hpp"


class pdf {
    public:
        virtual ~pdf() {}

        virtual __device__ float value(const linalg::vector3& direction) const = 0;
        virtual __device__ linalg::vector3 generate(curandState* random) const = 0;
};

static __device__ linalg::vector3 random_cosine_direction(curandState* random) {
    auto r1 = curand_uniform(random);
    auto r2 = curand_uniform(random);
    auto z = sqrt(1 - r2);

    auto phi = 2 * M_PI * r1;
    auto x = cos(phi) * sqrt(r2);
    auto y = sin(phi) * sqrt(r2);

    return {x, y, z};
}

class cosine_pdf : public pdf {
    public:
        __device__ cosine_pdf(const linalg::vector3& w) { uvw.build_from_w(w); }

        virtual __device__ float value(const linalg::vector3& direction) const override {
            auto cosine = linalg::dot(linalg::normalize(direction), uvw.w);
            return (cosine <= 0) ? 0 : cosine / M_PI;
        }

        virtual __device__ linalg::vector3 generate(curandState* random) const override {
            return uvw.local(random_cosine_direction(random));
        }

    public:
        onb uvw;
};


#endif //PDF_HPP