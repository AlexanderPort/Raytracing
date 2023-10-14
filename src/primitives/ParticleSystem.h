#ifndef PARTICLE_SYSTEM
#define PARTICLE_SYSTEM

#include "Particle.h"

#include <iostream>

class ParticleSystem {
    public:
        Vector4 position;
        int max_lifetime;
        int max_particles;
        int num_particles;
        Renderer* renderer;
        Particle** particles;

        ParticleSystem(Renderer* renderer, Vector4 position, 
                       int max_particles, int max_lifetime) {
            particles = new Particle*[max_particles];
            this->max_particles = max_particles;
            this->max_lifetime = max_lifetime;
            this->position = position;
            this->renderer = renderer;
            this->num_particles = 30;

            for (int i = 0; i < max_particles; i++)
                particles[i] = nullptr;
        }

        void update() {
            int num_particles = 0;
            for (int i = 0; i < max_particles; i++) {
                Particle* &particle = particles[i];
                if (particle == nullptr) {
                    if (num_particles > this->num_particles) continue;
                    particle = new Particle(position, Vector4(255, 0, 0, 1));
                    particle->pspeed = Vector4(rand() / 10000.0f, rand() / 10000.0f, rand() / 10000.0f, 0);
                } else { if (particle->update(max_lifetime)) particle = nullptr; }
                num_particles += 1;
            }

        }

        void draw() {
            for (int i = 0; i < max_particles; i++) {
                Particle* particle = particles[i];
                if (particle != nullptr) particle->draw(renderer);
            }
        }
};

#endif //PARTICLE_SYSTEM