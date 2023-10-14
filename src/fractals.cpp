/*
#include <iostream>
#include <vector>


#include <SDL.h>
#include <SDL_image.h>
#include <complex>
#include "rendering/Texture.h"



typedef std::complex<float> complex;
int ITERATIONS = 10;
int threshold = 4;
float power = 2;



inline int belong(complex& z, complex& c) {
    int iterations = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        z = std::pow(z, power) + c;
        iterations += 1;
        float real = z.real(), imag = z.imag();
        if (real * real + imag * imag > threshold) {
            return iterations;
        }
    }
    return iterations;
}


int main(int argc, char* argv[]) {
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    if (SDL_Init(SDL_INIT_VIDEO) == 0) {

        SDL_Window* window = SDL_CreateWindow(
                "window", 0, 0, 1000, 1000, SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN_DESKTOP);

        SDL_Surface* window_surface = SDL_GetWindowSurface(window);
        SDL_Renderer* renderer2D = SDL_CreateRenderer(
                window, -1, SDL_RENDERER_ACCELERATED);
        SDL_bool done = SDL_FALSE;
        float screen_width = window_surface->w;
        float screen_height = window_surface->h;
        float screen_h_width = screen_width / 2;
        float screen_h_height = screen_height / 2;
        uint* pixel_buffer = static_cast<uint*>(window_surface->pixels);
        std::string assets_path = "/home/alexander/Projects/CLionProjects/Graphics3D/src/assets/";
        std::string models_path = assets_path + "models/";
        std::string textures_path = assets_path + "textures/";
        Texture gradient_texture = Texture(textures_path + "gradient3.jpg", window_surface->format);
        uint* gradient_buffer = gradient_texture.buffer;


        SDL_Point* screen_point, cartesian_point;
        float C = 0;
        int max_color = 0xFFFFFF;
        float scale = 2;
        float dx = 0, dy = 0;
        bool change = true;
        float factor = screen_h_height / screen_h_width;
        while (!done) {
            SDL_Event event;
            SDL_FillRect(window_surface, nullptr, 0xFFFFFF);
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    done = SDL_TRUE;
                } else if (event.type == SDL_KEYDOWN) {
                    if (event.key.keysym.sym == SDLK_w) {
                        dy += 0.5;
                    } else if (event.key.keysym.sym == SDLK_s) {
                        dy -= 0.5;
                    } else if (event.key.keysym.sym == SDLK_a) {
                        dx -= 0.5;
                    } else if (event.key.keysym.sym == SDLK_d) {
                        dx += 0.5;
                    } else if (event.key.keysym.sym == SDLK_q) {
                        scale *= 1.1;
                    } else if (event.key.keysym.sym == SDLK_e) {
                        scale /= 1.1;
                    } else if (event.key.keysym.sym == SDLK_SPACE) {
                        change = !change;
                    } else if (event.key.keysym.sym == SDLK_z) {
                        if (ITERATIONS > 1) ITERATIONS -= 1;
                    } else if (event.key.keysym.sym == SDLK_x) {
                        ITERATIONS += 1;
                    } else if (event.key.keysym.sym == SDLK_c) {
                        threshold -= 1;
                    } else if (event.key.keysym.sym == SDLK_v) {
                        threshold += 1;
                    }
                } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                    dx += (event.motion.x - screen_h_width) / 1000;
                    dy += (screen_h_height - event.motion.y) / 1000;
                    scale *= 1.1;
                }
            }
#pragma omp parallel for
            for (int y = 0; y < int(screen_height); ++y) {
                float Y = 1 - 2 * y / screen_h_height;
                for (int x = 0; x < int(screen_width); ++x) {
                    float X = 2 * x / screen_h_width - 1;
                    std::complex<float> c = std::complex<float>(
                            (X + dx) / scale,(Y + dy) / scale * factor);
                    std::complex<float> z = std::complex<float>(0, 0);
                    //int iterations = belong(z, 2 * std::sin(c) + std::cos(c));
                    int iterations = belong(z, c);
                    int color = 0xFFFFFF * (float)iterations / ITERATIONS;
                    //int tx = gradient_texture.width * (float)iterations / ITERATIONS;
                    //uint color = gradient_buffer[tx + 0 * gradient_texture.width];
                    pixel_buffer[int(x + screen_width * y)] = color;
                }
            }
            SDL_UpdateWindowSurface(window);
            if (change) C -= 0.01;
            //max_color = rand() % 0xFFFFFF;

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