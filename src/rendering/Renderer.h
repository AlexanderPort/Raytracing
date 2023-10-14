//
// Created by alexander on 23.02.2021.
//

#ifndef RENDERER
#define RENDERER

#include "Camera.h"
#include "Rasterizer.h"
#include "../linalg/Vector4.h"
#include "../linalg/Matrix4x4.h"

class Renderer {
public:
    Camera camera;
    int width, height;
    int h_width, h_height;
    SDL_Surface* surface;
    SDL_Renderer* renderer;
    Rasterizer rasterizer{};
    SDL_PixelFormat* format{};
    Matrix4x4 mViewProjection;
    Matrix4x4 mScreenProjection;
    Matrix4x4 mView, mProjection, mScreen;
    Vector4 light = Vector4(0.0f, 0.0f, 0.0f);
    Renderer(SDL_Surface* surface, SDL_Renderer* renderer) {
        this->surface = surface;
        this->renderer = renderer;
        this->width = surface->w;
        this->height = surface->h;
        this->h_width = width / 2;
        this->h_height = height / 2;
        this->format = surface->format;
        this->camera = Camera(width, height);
        this->rasterizer = Rasterizer(surface);
    }
    void update() {
        mView = camera.view_matrix();
        mScreen = camera.screen_matrix();
        mProjection = camera.projection_matrix();
        mScreenProjection = mScreen * mProjection;
        mViewProjection = mScreenProjection * mView;
        memset(rasterizer.zbuffer, 0, rasterizer.size * sizeof(float));
    }
};

#endif //RENDERER
