//
// Created by alexander on 25.02.2021.
//

#ifndef TRIANGLE
#define TRIANGLE

#include "../linalg/Vector2.h"
#include "../linalg/Vector3.h"
#include "../linalg/Vector4.h"

#include <algorithm>

class Triangle {
public:
    Vector4 *v1, *v2, *v3;
    Vector4 *n1, *n2, *n3;
    Vector2 *uv1, *uv2, *uv3;
    Triangle() = default;
    Triangle(Vector4* v1, Vector4* v2, Vector4* v3,
             Vector4* n1, Vector4* n2, Vector4* n3,
             Vector2* uv1, Vector2* uv2, Vector2* uv3) {
        this->v1 = v1; this->v2 = v2; this->v3 = v3;
        this->n1 = n1; this->n2 = n2; this->n3 = n3;
        this->uv1 = uv1; this->uv2 = uv2; this->uv3 = uv3;
    }
    Vector4 normal() {
        float x21 = v2->x - v1->x;
        float y21 = v2->y - v1->y;
        float z21 = v2->z - v1->z;

        float x32 = v3->x - v2->x;
        float y32 = v3->y - v2->y;
        float z32 = v3->z - v2->z;

        float x = y21 * z32 - y32 * z21;
	    float y = z21 * x32 - z32 * x21;
	    float z = x21 * y32 - x32 * y21;

        return {x, y, z, 1};
    }
    static Vector3 normal(const Vector4& v1, 
                          const Vector4& v2, 
                          const Vector4& v3) {
        
        float x21 = v2.x - v1.x;
        float y21 = v2.y - v1.y;
        float z21 = v2.z - v1.z;

        float x32 = v3.x - v2.x;
        float y32 = v3.y - v2.y;
        float z32 = v3.z - v2.z;

        float x = y21 * z32 - y32 * z21;
	    float y = z21 * x32 - z32 * x21;
	    float z = x21 * y32 - x32 * y21;

        return {x, y, z};
    }

    static bool ccw(Vector4& v1, Vector4& v2, Vector4& v3) {
        float dx21 = v2.x - v1.x;
        float dy21 = v2.y - v1.y;
        float dx31 = v3.x - v1.x;
        float dy31 = v3.y - v1.y;
        return (dx21 * dy31 - dy21 * dx31) > 0;
    }
};

#endif //TRIANGLE
