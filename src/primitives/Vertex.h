#ifndef VERTEX
#define VERTEX

#include "../linalg/Vector2.h"
#include "../linalg/Vector3.h"
#include "../linalg/Vector4.h"

class Vertex {
    public:
        Vector4 *coord, *normal, *texture;

        Vertex(Vector4 *coord, Vector4 *normal, Vector4 *texture) {
            this->coord = coord;
            this->normal = normal;
            this->texture = texture;
        }


};

#endif //VERTEX