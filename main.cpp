#include "raytrace.cuh"

#include <iostream>

#define SDL_main main
#include <SDL.h>
#include <SDL_image.h>

#include "ray.hpp"
#include "linalg.hpp"
#include "camera.hpp"

int main(int argc, char* argv[]) {
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    if (SDL_Init(SDL_INIT_VIDEO) == 0) {

        SDL_Window* window = SDL_CreateWindow(
                "window",  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1024, 768,
                SDL_WINDOW_VULKAN | SDL_WINDOW_ALLOW_HIGHDPI);

        SDL_Surface* window_surface = SDL_GetWindowSurface(window);
        int screen_width = window_surface->w; int screen_height = window_surface->h;
        unsigned int* screen_buffer = static_cast<unsigned int*>(window_surface->pixels);
        SDL_Renderer* renderer2D = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

        SDL_Surface* environment_surface = IMG_Load("C:/Projects/RayTracing/mars.jpg");
        environment_surface = SDL_ConvertSurface(environment_surface, window_surface->format, 0);
        unsigned int* environment_buffer = static_cast<unsigned int*>(environment_surface->pixels);
        int environment_width = environment_surface->w; int environment_height = environment_surface->h;
        

        Camera camera = Camera(screen_width, screen_height);

        SDL_bool done = SDL_FALSE; float c = 0.01f; double s = 1.0f;

        bool MOUSE_BUTTON_DOWN = false;
        double dx = 0;    double dy = 0;
        double mouse_x = 0, mouse_y = 0;
        double min_x = -1.0f, max_x = 1.0f;
        double min_y = -1.0f, max_y = 1.0f;

        int frames = 0;

        while (!done) {
            SDL_Event event; frames += 1;
            //SDL_FillRect(window_surface, nullptr, 0x000000);
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    done = SDL_TRUE;
                } else if (event.type == SDL_KEYDOWN) {
                    if (event.key.keysym.sym == SDLK_1) {
                        c += 0.001f;
                    } else if (event.key.keysym.sym == SDLK_2) {
                        c -= 0.001f;
                    } else if (event.key.keysym.sym == SDLK_3) {
                        s *= 1.1f;
                        min_x *= 1.1f; max_x *= 1.1f;
                        min_y *= 1.1f; max_y *= 1.1f;
                    } else if (event.key.keysym.sym == SDLK_4) {
                        s /= 1.1f;
                        min_x /= 1.1f; max_x /= 1.1f;
                        min_y /= 1.1f; max_y /= 1.1f;
                    }
                } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                    MOUSE_BUTTON_DOWN = true;
                    mouse_x = event.motion.x;
                    mouse_y = event.motion.y;
                } else if (event.type == SDL_MOUSEBUTTONUP) {
                    MOUSE_BUTTON_DOWN = false;
                } else if (event.type == SDL_MOUSEMOTION) {
                    if (MOUSE_BUTTON_DOWN) {
                        dx += (mouse_x - event.motion.x) * s;
                        dy += (mouse_y - event.motion.y) * s;
                        mouse_x = event.motion.x; 
                        mouse_y = event.motion.y; 
                    }
                }
                if (camera.control(event)) { frames = 1; }
            }
            raytracing::raytrace(screen_buffer, screen_width, screen_height, environment_buffer, 
                                 environment_width, environment_height, 1.0f / frames, camera); 
            SDL_UpdateWindowSurface(window);
        }
        
        if (renderer2D) {
            SDL_DestroyRenderer(renderer2D);
        }
        
        if (window) {
            SDL_DestroyWindow(window);
        }
        SDL_Quit();
    }
    raytracing::free();
    return 0;
}