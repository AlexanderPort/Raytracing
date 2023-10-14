#ifndef LIGHT_HPP
#define LIGHT_HPP

#include "linalg.hpp"

class light {
    public:
        linalg::vector3 position;
        linalg::vector3 intensity;

        __device__ light(const linalg::vector3& position, const linalg::vector3& intensity) {
            this->position = position;
            this->intensity = intensity;
        }

};

#endif //LIGHT_HPP