//
// Created by alexander on 23.02.2021.
//

#ifndef RASTERIZER
#define RASTERIZER

#include <SDL.h>
#include "../linalg/Vector2.h"
#include "../linalg/Vector4.h"
#include "../linalg/Matrix4x4.h"
#include "../primitives/Triangle.h"
#include "Texture.h"
#include <algorithm>


class Rasterizer {
public:
    int size;
    float* zbuffer;
    int width, height;
    unsigned int* cbuffer;
    SDL_PixelFormat* format;
    Rasterizer() = default;
    explicit Rasterizer(SDL_Surface* surface) {
        this->width = surface->w;
        this->height = surface->h;
        this->format = surface->format;
        this->size = width * height;
        this->zbuffer = new float [size];
        this->cbuffer = static_cast<unsigned int*>(surface->pixels);
    }
    void rasterize(Vector4& V1, Vector4& V2, Vector4& V3,
                   Vector2& UV1, Vector2& UV2, Vector2& UV3,
                   Texture* texture) {
        float screen_width = (float)width;
        float screen_height = (float)height;
        float texture_width = (float)texture->width;
        float texture_height = (float)texture->height;

        float x1 = V1.x, x2 = V2.x, x3 = V3.x;
        float y1 = V1.y, y2 = V2.y, y3 = V3.y;
        float z1 = V1.w, z2 = V2.w, z3 = V3.w;

        float u1 = UV1.x, v1 = UV1.y;
        float u2 = UV2.x, v2 = UV2.y;
        float u3 = UV3.x, v3 = UV3.y;

        bool in_x1 = 0 < x1 && x1 < screen_width;
        bool in_x2 = 0 < x2 && x2 < screen_width;
        bool in_x3 = 0 < x3 && x3 < screen_width;

        bool in_y1 = 0 < y1 && y1 < screen_height;
        bool in_y2 = 0 < y2 && y2 < screen_height;
        bool in_y3 = 0 < y3 && y3 < screen_height;

        if (!(in_x1 && in_x2 && in_x3 && in_y1 && in_y2 && in_y3)) return;

        //x1 = std::max(std::min(x1, screen_width), 0.0f);
        //x2 = std::max(std::min(x2, screen_width), 0.0f);
        //x3 = std::max(std::min(x3, screen_width), 0.0f);

        //y1 = std::max(std::min(y1, screen_height), 0.0f);
        //y2 = std::max(std::min(y2, screen_height), 0.0f);
        //y3 = std::max(std::min(y3, screen_height), 0.0f);

        int max_x = int(std::max({ x1, x2, x3 }));
        int min_x = int(std::min({ x1, x2, x3 }));
        int max_y = int(std::max({ y1, y2, y3 }));
        int min_y = int(std::min({ y1, y2, y3 }));

        max_x = std::min(max_x + 1, width);
        min_x = std::max(min_x - 1, 0);
        max_y = std::min(max_y + 1, height);
        min_y = std::max(min_y - 1, 0);

        float area = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1);
        if (area != 0 && z1 != 0 && z2 != 0 && z3 != 0) {
            z1 = 1 / z1 / area; z2 = 1 / z2 / area; z3 = 1 / z3 / area;

            u1 = u1 * z1 * texture_width; v1 = v1 * z1 * texture_height;
            u2 = u2 * z2 * texture_width; v2 = v2 * z2 * texture_height;
            u3 = u3 * z3 * texture_width; v3 = v3 * z3 * texture_height;

            float dx1 = x2 - x3, dx2 = x3 - x1, dx3 = x1 - x2;
            float dy1 = y3 - y2, dy2 = y1 - y3, dy3 = y2 - y1;

            float fmin_x = (float)min_x, fmin_y = (float)min_y;
            float W1 = (fmin_x - x2) * (y3 - y2) - (fmin_y - y2) * (x3 - x2);
            float W2 = (fmin_x - x3) * (y1 - y3) - (fmin_y - y3) * (x1 - x3);
            float W3 = (fmin_x - x1) * (y2 - y1) - (fmin_y - y1) * (x2 - x1);

            for (int y = min_y; y < max_y; y++) {
                float w1 = W1, w2 = W2, w3 = W3;
                for (int x = min_x; x < max_x; x++) {
                    if (w1 < 0 && w2 < 0 && w3 < 0) {
                        unsigned int index = x + y * width;
                        float z = w1 * z1 + w2 * z2 + w3 * z3;
                        if (z > zbuffer[index]) {
                            zbuffer[index] = z;
                            int u = int((w1 * u1 + w2 * u2 + w3 * u3) / z);
                            int v = int((w1 * v1 + w2 * v2 + w3 * v3) / z);
                            //u = std::max(std::min((float)u, texture_width), 0.0f);
                            //v = std::max(std::min((float)v, texture_height), 0.0f);
                            cbuffer[index] = texture->buffer[int(u + v * texture_width)];
                        }
                    }
                    w1 += dy1; w2 += dy2; w3 += dy3;
                }
                W1 += dx1; W2 += dx2; W3 += dx3;
            }
        }
    }

    void rasterize(Vector4& V1, Vector4& V2, Vector4& V3,
                   Vector2& UV1, Vector2& UV2, Vector2& UV3,
                   Vector3& diffuse1, Vector3& diffuse2, Vector3& diffuse3,
                   Vector3& specular1, Vector3& specular2, Vector3& specular3,
                   Texture* texture) {
        float screen_width = (float)width;
        float screen_height = (float)height;
        float texture_width = (float)texture->width;
        float texture_height = (float)texture->height;

        float x1 = V1.x, x2 = V2.x, x3 = V3.x;
        float y1 = V1.y, y2 = V2.y, y3 = V3.y;
        float z1 = V1.w, z2 = V2.w, z3 = V3.w;

        float u1 = UV1.x, v1 = UV1.y;
        float u2 = UV2.x, v2 = UV2.y;
        float u3 = UV3.x, v3 = UV3.y;

        bool in_x1 = 0 < x1 && x1 < screen_width;
        bool in_x2 = 0 < x2 && x2 < screen_width;
        bool in_x3 = 0 < x3 && x3 < screen_width;

        bool in_y1 = 0 < y1 && y1 < screen_height;
        bool in_y2 = 0 < y2 && y2 < screen_height;
        bool in_y3 = 0 < y3 && y3 < screen_height;

        if (!(in_x1 && in_x2 && in_x3 && in_y1 && in_y2 && in_y3)) return;

        //x1 = std::max(std::min(x1, screen_width), 0.0f);
        //x2 = std::max(std::min(x2, screen_width), 0.0f);
        //x3 = std::max(std::min(x3, screen_width), 0.0f);

        //y1 = std::max(std::min(y1, screen_height), 0.0f);
        //y2 = std::max(std::min(y2, screen_height), 0.0f);
        //y3 = std::max(std::min(y3, screen_height), 0.0f);

        int max_x = int(std::max({ x1, x2, x3 }));
        int min_x = int(std::min({ x1, x2, x3 }));
        int max_y = int(std::max({ y1, y2, y3 }));
        int min_y = int(std::min({ y1, y2, y3 }));

        max_x = std::min(max_x + 1, width);
        min_x = std::max(min_x - 1, 0);
        max_y = std::min(max_y + 1, height);
        min_y = std::max(min_y - 1, 0);

        float area = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1);
        if (area != 0 && z1 != 0 && z2 != 0 && z3 != 0) {
            z1 = 1 / z1 / area; z2 = 1 / z2 / area; z3 = 1 / z3 / area;

            u1 = u1 * z1 * texture_width; v1 = v1 * z1 * texture_height;
            u2 = u2 * z2 * texture_width; v2 = v2 * z2 * texture_height;
            u3 = u3 * z3 * texture_width; v3 = v3 * z3 * texture_height;

            diffuse1 = diffuse1 * z1; specular1 = specular1 * z1;
            diffuse2 = diffuse2 * z2; specular2 = specular2 * z2;
            diffuse3 = diffuse3 * z3; specular3 = specular3 * z3;

            float dx1 = x2 - x3, dx2 = x3 - x1, dx3 = x1 - x2;
            float dy1 = y3 - y2, dy2 = y1 - y3, dy3 = y2 - y1;

            float fmin_x = (float)min_x, fmin_y = (float)min_y;
            float W1 = (fmin_x - x2) * (y3 - y2) - (fmin_y - y2) * (x3 - x2);
            float W2 = (fmin_x - x3) * (y1 - y3) - (fmin_y - y3) * (x1 - x3);
            float W3 = (fmin_x - x1) * (y2 - y1) - (fmin_y - y1) * (x2 - x1);

            for (int y = min_y; y < max_y; y++) {
                float w1 = W1, w2 = W2, w3 = W3;
                for (int x = min_x; x < max_x; x++) {
                    if (w1 < 0 && w2 < 0 && w3 < 0) {
                        unsigned int index = x + y * width;
                        float z = w1 * z1 + w2 * z2 + w3 * z3;
                        if (z > zbuffer[index]) {
                            zbuffer[index] = z;
                            int u = int((w1 * u1 + w2 * u2 + w3 * u3) / z);
                            int v = int((w1 * v1 + w2 * v2 + w3 * v3) / z);
                            Vector3 diffuse = (w1 * diffuse1 + w2 * diffuse2 + w3 * diffuse3) / z;
                            Vector3 specular = (w1 * specular1 + w2 * specular2 + w3 * specular3) / z;
                            unsigned int pixel = texture->buffer[int(u + v * texture_width)];
                            unsigned char r, g, b; SDL_GetRGB(pixel, format, &r, &g, &b);
                            Vector3 ambient = Vector3(0, 0, 100);
                            float R = ambient.x + diffuse.x * 255 + specular.x * 255;
                            float G = ambient.y + diffuse.y * 255 + specular.y * 255;
                            float B = ambient.z + diffuse.z * 255 + specular.z * 255;
                            if (R > 255) R = 255; if (G > 255) G = 255; if (B > 255) B = 255;
                            //u = std::max(std::min((float)u, texture_width), 0.0f);
                            //v = std::max(std::min((float)v, texture_height), 0.0f);
                            cbuffer[index] = SDL_MapRGB(format, R, G, B);
                        }
                    }
                    w1 += dy1; w2 += dy2; w3 += dy3;
                }
                W1 += dx1; W2 += dx2; W3 += dx3;
            }
        }
    }

    void interpolate(float x1, float y1, float x2, float y2, float* array, int* size) {
        if (x1 == x2) { *size = 1; array = new float[1]{y1}; }
        
        *size = int(x2) - int(x1);
        array = new float [*size]; int i = 0;
        float a = (y2 - y1) / (x2 - x1), y = y1;

        for (int x = int(x1); x < int(x2); x++) {
            array[i] = y; i += 1; y += a;
        }
    }

    void interpolate(float x1, float y1, float x2, float y2, float* array) {
        if (x1 == x2) { array[0] = y1; }
        
        int i = 0;
        float a = (y2 - y1) / (x2 - x1), y = y1;

        for (int x = int(x1); x < int(x2); x++) {
            array[i] = y; i += 1; y += a;
        }
    }

    void conv2d(float* filter, int w, int h) {
        int new_width = width - w + 1;
        int new_height = height - h + 1;
        unsigned int* buffer = new unsigned int [new_width * new_height];
        #pragma omp parallel for
        for (int y = 0; y < height - h; y += 1) {
            for (int x = 0; x < width - w; x += 1) {
                float R = 0, G = 0, B = 0;
                for (int i = y; i < y + h; i++) {
                    for (int j = x; j < x + w; j++) {
                        unsigned char r, g, b;
                        int p = cbuffer[j + i * width];
                        SDL_GetRGB(p, format, &r, &g, &b);
                        float f = filter[(j - x) + (i - y) * w]; 
                        R += f * r;    G += f * g;    B += f * b;
                    }
                }
                buffer[x + y * new_width] = SDL_MapRGB(format, (uint8_t)R, (uint8_t)G, (uint8_t)B);;
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < new_height; i++) {
            for (int j = 0; j < new_width; j++) {
                cbuffer[j + i * width] = buffer[j + i * new_width];
            }
        }
        delete[] buffer;
    }

    void conv3d(float* filter, int w, int h) {
        int new_width = width - w + 1;
        int new_height = height - h + 1;
        unsigned int* buffer = new unsigned int [new_width * new_height];
        #pragma omp parallel for
        for (int y = 0; y < height - h; y += 1) {
            for (int x = 0; x < width - w; x += 1) {
                float R = 0, G = 0, B = 0;
                for (int i = y; i < y + h; i++) {
                    for (int j = x; j < x + w; j++) {
                        unsigned char r, g, b;
                        int p = cbuffer[j + i * width];
                        SDL_GetRGB(p, format, &r, &g, &b);
                        float f = filter[(j - x) + (i - y) * w]; 
                        R += f * r;    G += f * g;    B += f * b;
                    }
                }
                uint8_t color = (uint8_t)(R + B + G);
                buffer[x + y * new_width] = SDL_MapRGB(format, color, color, color);;
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < new_height; i++) {
            for (int j = 0; j < new_width; j++) {
                cbuffer[j + i * width] = buffer[j + i * new_width];
            }
        }
        delete[] buffer;
    }

};

#endif //RASTERIZER
