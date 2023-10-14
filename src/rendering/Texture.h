//
// Created by alexander on 24.02.2021.
//

#ifndef TEXTURE
#define TEXTURE
#include <vector>
#include <string>
#include <SDL.h>
#include <SDL_image.h>

class Texture {
public:
    unsigned int* buffer;
    std::string filename;
    SDL_Surface* surface;
    int width, height, size;
    SDL_PixelFormat* format;
    Texture(const std::string& filename, SDL_PixelFormat* format) {
        std::cout << "Loading texture... from file " << filename << std::endl;
        this->format = format;
        this->filename = filename;
        this->surface = IMG_Load(filename.c_str());
        this->width = surface->w;
        this->height = surface->h;
        this->size = width * height;
        this->surface = SDL_ConvertSurface(surface, format, 0);
        this->buffer = static_cast<unsigned int*>(surface->pixels);
        std::cout << "Sizes of texture: " << width << " " << height << std::endl;
    }
};

#endif //TEXTURE
