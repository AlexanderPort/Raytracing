//
// Created by alexander on 08.03.2021.
//

#ifndef RAY
#define RAY
#include "../linalg/Vector3.h"
#include "../linalg/Vector4.h"


class Ray {
public:
    Vector3 origin, direction;
    Ray(Vector3& origin, Vector3& direction) {
        this->origin = origin;
        this->direction = direction;
    }

};

#endif //RAY
