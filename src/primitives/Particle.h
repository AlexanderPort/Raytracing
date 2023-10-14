#ifndef PARTICLE
#define PARTICLE

#include "../linalg/Vector3.h"
#include "../linalg/Vector4.h"

#include "../rendering/Renderer.h"

class Particle {
    public:
        Vector4 position;
        Vector4 pspeed;
        Vector4 cspeed;
        Vector4 color;
        int lifetime = 0;

        Particle() = default;

        Particle(Vector4 position, Vector4 color) {
            this->position = position;
            this->color = color;
        }

        bool update(int max_lifetime) {
            if (lifetime > max_lifetime) return true;

            position.x += pspeed.x;
            position.y += pspeed.y;
            position.z += pspeed.z;

            color.x += cspeed.x;
            color.y += cspeed.y;
            color.z += cspeed.z;

            lifetime += 1;
        }

        void draw(Renderer* renderer3D) {
            Vector4 pposition = renderer3D->mScreenProjection * position;
            pposition.x /= pposition.w; pposition.y /= pposition.w;
            if (0 < pposition.x && pposition.x < renderer3D->width && 
                0 < pposition.y && pposition.y < renderer3D->height) {
                renderer3D->draw_point(pposition.x, pposition.y, 
                    color.x, color.y, color.z, color.w);
            }
           
        }
};

#endif //PARTICLE