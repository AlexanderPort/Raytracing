/*
#include <iostream>
#include <vector>


#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <complex>
#include "rendering/Texture.h"
#include "rendering/Renderer.h"
#include "primitives/Object3D.h"
#include "primitives/Instance3D.h"
#include "linalg/Vector3.h"
#include <algorithm>
#include <CL/cl.h>


int ITERATIONS = 300;
float MINIMUM_HIT_DISTANCE = 0.001;
float MAXIMUM_TRACE_DISTANCE = 100;

int threshold = 4;
float power = 1;


class Shape {
public:
    Vector3 position;

};




class Sphere {
public:
    float radius;
    Vector3 position;
    Vector3 rotation = Vector3(0, 0, 0);
    Matrix4x4 rotation_matrix = Matrix4x4::identity();
    Sphere(Vector3& position, float radius) {
        this->position = position;
        this->radius = radius;
    }
    Sphere(Vector3 position, float radius) {
        this->position = position;
        this->radius = radius;
    }
    Sphere(Vector3& position) {
        this->position = position;
        this->radius = 1;
    }
    inline float distance(Vector3& point) {

        float x = point.x - position.x;
        float y = point.y - position.y;
        float z = point.z - position.z;
        rotation_matrix.multiply(&x, &y, &z);
        //x = std::sin(x);
        //y = std::sin(y);
        //z = std::sin(z);

        float sphere = (std::sqrt(x*x+y*y+z*z)) - radius / 2 - 0.1;

        float cube = std::max({std::abs(x),
                               std::abs(y),
                               std::abs(z)}) - radius / 2;

        return std::max(-sphere, cube);


        float tor = std::sqrt(std::pow(std::sqrt(x*x + z*z) - 3, 2) + y*y) - 1;

        return tor;


        float x = point.x - position.x;
        float y = point.y - position.y;
        float z = point.z - position.z;
        return std::min(
                {std::sqrt(x*x + std::pow((y - 0.27), 2) + z*z),
                std::sqrt(x*x + 2.5*y*y + z*z) - 0.4,
                std::sqrt(std::pow(std::sqrt(x*x + z*z) - 0.3, 2) + std::pow(y - 0.18, 2)) - 0.02,
                std::max({x + y - 0.7, -y + 0.09})}
                );

    }
};

std::pair<Sphere, float> nearest(std::vector<Sphere>& shapes, Vector3 point) {
    Sphere SHAPE = shapes[0];
    float DISTANCE = SHAPE.distance(point);
    for (int i = 1; i < shapes.size(); ++i) {
        Sphere shape = shapes[i];
        float distance = shape.distance(point);;
        if (DISTANCE > distance) {
            SHAPE = shape;

            DISTANCE = distance;
        }
    }
    return std::pair<Sphere, float>(SHAPE, DISTANCE);
}

inline int RGBtoHEX(int r, int g, int b)
{
    return (r<<16) | (g<<8) | b;
}
int* HEXtoRGB(int hex) {
    int* rgb = new int[3];
    rgb[0] = hex >> 16 & 0xFF;
    rgb[1] = hex >> 8 & 0xFF;
    rgb[2] = hex & 0xFF;
    return rgb;
}

inline uint raymarch(std::vector<Sphere>& shapes, Vector3& origin, Vector3& direction) {
    float DISTANCE = 0;

    for (int i = 0; i < ITERATIONS; ++i) {
        Vector3 current = origin + DISTANCE * direction;
        std::pair<Sphere, float> pair = nearest(shapes, current);
        Sphere shape = pair.first;
        float distance = pair.second;
        //std::cout << distance << std::endl;
        float d = float(i + 1) / ITERATIONS;
        if (distance < MINIMUM_HIT_DISTANCE) {
            int r = 255 * d;
            int g = 0;
            int b = 0;
            return RGBtoHEX(r * 5, g, b);

            //return 256 * (1 - float(i) / ITERATIONS);
        }
        if (DISTANCE > MAXIMUM_TRACE_DISTANCE) {
            int r = 255 * d;
            int g = 0;
            int b = 0;
            return RGBtoHEX(r * 5, g, b);
        }
        DISTANCE += distance;
    }
    return 0xFFFFFF;
}


int main(int argc, char* argv[]) {
    //SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");


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
        std::vector<Sphere> shapes = std::vector<Sphere>();


        std::string assets_path = "/home/alexander/Projects/CLionProjects/Graphics3D/src/assets/";
        std::string models_path = assets_path + "models/";
        std::string textures_path = assets_path + "textures/";

        Texture cube_texture = Texture(textures_path + "space.jpg", renderer3D.format);
        Object3D cube_object = Object3D::load(models_path + "cube.obj");
        cube_object.texture = &cube_texture;
        Instance3D cube = Instance3D(&cube_object);


        float far = renderer3D.camera.far_plane;
        float near = renderer3D.camera.near_plane;
        float ndc_depth = far - near;
        float ndc_summa = far + near;
        float aspect_y = 1;
        float aspect_x = 1;
        float factor_x = std::tan(renderer3D.camera.h_fov / 2);
        float factor_y = std::tan(renderer3D.camera.v_fov / 2);

        shapes.push_back(Sphere(Vector3(0, 0, 0), 1));
        shapes.push_back(Sphere(Vector3(3, 3, 3), 1));


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

            //Matrix4x4 cameraToWorld = (renderer3D.mProjection * renderer3D.mView).inverse();
            Vector3 up = renderer3D.camera.up;
            Vector3 right = renderer3D.camera.right;
            Vector3 forward = renderer3D.camera.forward;

#pragma omp parallel for
            for (int y = 0; y < int(screen_height); ++y) {
                float ndc_y = (1 - 2 * y / screen_height) * aspect_y * factor_y;
                for (int x = 0; x < int(screen_width); ++x) {
                    float ndc_x = (2 * x / screen_width - 1) * aspect_x * factor_x;
                    Vector3 origin = renderer3D.camera.position;
                    Vector3 direction = ndc_x * right + ndc_y * up + forward;
                    direction = direction.normalize();
                    cbuffer[int(x + y * screen_width)] = raymarch(shapes, origin, direction);
                }
            }
            //std::cout << "UPDATE" << std::endl;
            //cube.render(&renderer3D);
            //cube.rotation += 1;
            SDL_UpdateWindowSurface(window);

            //ITERATIONS = (ITERATIONS + 1) % 1000 + 1;


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