/*
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include "linalg/Vector2.h"
#include "linalg/Vector3.h"
#include "linalg/Vector4.h"
#include "linalg/Matrix4x4.h"
#include "rendering/Texture.h"
#include "rendering/Renderer.h"
#include "primitives/Ray.h"
#include "primitives/Object3D.h"
#include "primitives/Instance3D.h"


class Shape {
public:
    Vector3 position;
    Vector3 rotation;
    Vector3 scale;
    virtual float intersect(Ray& ray) {
        return -1;
    }
};

class Sphere : public Shape {
public:
    float radius;
    Sphere() {
        this->position = Vector3(0, 0, 0);
        this->rotation = Vector3(0, 0, 0);
        this->scale = Vector3(1, 1, 1);
        this->radius = 1;
    }
    Sphere(Vector3 position) {
        this->position = position;
        this->rotation = Vector3(0, 0, 0);
        this->scale = Vector3(1, 1, 1);
        this->radius = 1;
    }
    Sphere(Vector3& position) {
        this->position = position;
        this->rotation = Vector3(0, 0, 0);
        this->scale = Vector3(1, 1, 1);
        this->radius = 1;
    }
    float intersect(Ray& ray) {
        Vector3 oc = ray.origin - position;
        float a = ray.direction.squared_length();
        float b = 2 * (oc * ray.direction).sum();
        float c = oc.squared_length() - radius * radius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) { return -1; }
        else { return (-b - std::sqrt(discriminant)) / a / 2; }
    }

};


int main(int argc, char* argv[]) {
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");

    if (SDL_Init(SDL_INIT_VIDEO) == 0) {

        SDL_Window* window = SDL_CreateWindow(
                "window", 0, 0, 1500, 1000, SDL_WINDOW_OPENGL);

        SDL_Surface* window_surface = SDL_GetWindowSurface(window);
        SDL_Renderer* renderer2D = SDL_CreateRenderer(
                window, -1, SDL_RENDERER_ACCELERATED);
        SDL_bool done = SDL_FALSE;
        float screen_width = window_surface->w;
        float screen_height = window_surface->h;
        float screen_h_width = screen_width / 2;
        float screen_h_height = screen_height / 2;

        Renderer renderer3D = Renderer(window_surface, renderer2D);
        renderer3D.camera.position = Vector4(0, 0, 0, 1);
        uint* cbuffer = renderer3D.graphics.cbuffer;

        std::string assets_path = "/home/alexander/Projects/CLionProjects/Graphics3D/src/assets/";
        std::string models_path = assets_path + "models/";
        std::string textures_path = assets_path + "textures/";


        //Texture cube_texture = Texture(textures_path + "space.jpg", renderer3D.format);
        //Object3D cube_object = Object3D::load(models_path + "cube.obj");
        //cube_object.texture = &cube_texture;
        //Instance3D cube = Instance3D(&cube_object);


        float aspect_y = 1; float aspect_x = 1;
        float far = renderer3D.camera.far_plane;
        float near = renderer3D.camera.near_plane;
        float factor_x = std::tan(renderer3D.camera.h_fov / 2);
        float factor_y = std::tan(renderer3D.camera.v_fov / 2);
        Sphere sphere = Sphere(Vector3(0, 5, 5));

        while (!done) {
            SDL_Event event;
            //SDL_FillRect(window_surface, nullptr, 0xFFFFFF);
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    done = SDL_TRUE;
                }
                renderer3D.camera.control(event);
            }
            renderer3D.update();

            Vector3 up = renderer3D.camera.up;
            Vector3 right = renderer3D.camera.right;
            Vector3 forward = renderer3D.camera.forward;
            Vector3 origin = renderer3D.camera.position;
#pragma omp parallel for
            for (int y = 0; y < int(screen_height); ++y) {
                float ndc_y = (1 - 2 * (y + 0.5) / screen_height) * factor_y;
                for (int x = 0; x < int(screen_width); ++x) {
                    float ndc_x = (2 * (x + 0.5) / screen_width - 1) * factor_x;
                    Vector3 direction = ndc_x * right + ndc_y * up + forward;
                    direction.normalize(); Ray ray = Ray(origin, direction);
                    float intersection = sphere.intersect(ray);
                    if (intersection != -1) {
                        float max_length = (ray.origin - sphere.position).length() + sphere.radius;
                        float length = (ray.direction * intersection).length();
                        cbuffer[int(x + y * screen_width)] = 255 * length;
                    } else {
                        cbuffer[int(x + y * screen_width)] = 0xFFFFFF;
                    }
                }
            }
            SDL_UpdateWindowSurface(window);
        }
        if (renderer2D) {
            SDL_DestroyRenderer(renderer2D);
        }
        if (window) {
            SDL_DestroyWindow(window);
        }
    }
    SDL_Quit();
    return 0;
}
 */