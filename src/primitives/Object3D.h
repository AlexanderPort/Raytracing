//
// Created by alexander on 23.02.2021.
//

#ifndef OBJECT3D
#define OBJECT3D
#include "../linalg/Vector4.h"
#include "../linalg/Matrix4x4.h"
#include "../rendering/Texture.h"
#include "Triangle.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <iostream>



class Object3D {
public:
    // triangle data format:
    // float buffer: first 9 elements - vertices coords
    // if texture coords exist: next 6 elements - texture coords
    // if normal coords exist: next 9 elements - normal coords

    Triangle** buffer{};
    Texture* texture{};
    int buffer_size;
    Object3D(Triangle** buffer, int buffer_size) {
        this->buffer = buffer;
        this->texture = nullptr;
        this->buffer_size = buffer_size;
    }
    Object3D(Triangle** buffer, int buffer_size, Texture* texture) {
        this->buffer = buffer;
        this->texture = texture;
        this->buffer_size = buffer_size;
    }

    static Object3D load(const std::string& filename) {
        std::ifstream in(filename, std::ios::in);
        if (!in)
        {
            std::cerr << "Cannot open " << filename << std::endl;
            exit(1);
        }

        std::vector<Vector4*> faceCoords = std::vector<Vector4*>();
        std::vector<Vector4*> normalCoords = std::vector<Vector4*>();
        std::vector<Vector2*> textureCoords = std::vector<Vector2*>();
        std::vector<unsigned int> faceIndexes = std::vector<unsigned int>();
        std::vector<unsigned int> normalIndexes = std::vector<unsigned int>();
        std::vector<unsigned int> textureIndexes = std::vector<unsigned int>();

        std::string line;
        while (std::getline(in, line)) {

            if (line.substr(0, 2) == "v ") {
                std::istringstream s(line.substr(2));
                float x, y, z; s >> x; s >> y; s >> z;
                //std::cout << x << " " << y << " " << z << std::endl;
                faceCoords.push_back(new Vector4(x, y, z, 1));
            }
            else if (line.substr(0, 2) == "vn") {
                std::istringstream s(line.substr(2));
                float x, y, z; s >> x; s >> y; s >> z;
                //std::cout << x << " " << y << " " << z << std::endl;
                normalCoords.push_back(new Vector4(x, y, z, 1));
            }
            else if (line.substr(0, 2) == "vt") {
                std::istringstream s(line.substr(2));
                float u, v; s >> u; s >> v;
                //std::cout << u << " " << v << std::endl;
                textureCoords.push_back(new Vector2(u, 1 - v));
            }
            else if (line.substr(0, 2) == "f ") {
                int spaces = std::count(std::begin(line), std::end(line), ' ');
                int slashes = std::count(std::begin(line), std::end(line), '/');
                if (spaces == 4) {
                    if (slashes == 4) {
                        int f1, t1;
                        int f2, t2;
                        int f3, t3;
                        int f4, t4;
                        const char* chars = line.c_str();
                        std::sscanf(chars, "f %i/%i %i/%i %i/%i %i/%i",
                                    &f1, &t1, &f2, &t2, &f3, &t3, &f4, &t4);

                        faceIndexes.push_back(f1 - 1);
                        faceIndexes.push_back(f2 - 1);
                        faceIndexes.push_back(f3 - 1);
                        faceIndexes.push_back(f1 - 1);
                        faceIndexes.push_back(f3 - 1);
                        faceIndexes.push_back(f4 - 1);
                        
                        textureIndexes.push_back(t1 - 1);
                        textureIndexes.push_back(t2 - 1);
                        textureIndexes.push_back(t3 - 1);
                        textureIndexes.push_back(t1 - 1);
                        textureIndexes.push_back(t3 - 1);
                        textureIndexes.push_back(t4 - 1);
                    } else if (slashes == 8) {
                        int f1, t1, n1;
                        int f2, t2, n2;
                        int f3, t3, n3;
                        int f4, t4, n4;
                        const char* chars = line.c_str();
                        std::sscanf(chars, "f %i/%i/%i %i/%i/%i %i/%i/%i %i/%i/%i",
                                    &f1, &t1, &n1, &f2, &t2, &n2, &f3, &t3, &n3, &f4, &t4, &n4);

                        faceIndexes.push_back(f1 - 1);
                        faceIndexes.push_back(f2 - 1);
                        faceIndexes.push_back(f3 - 1);
                        faceIndexes.push_back(f1 - 1);
                        faceIndexes.push_back(f3 - 1);
                        faceIndexes.push_back(f4 - 1);

                        normalIndexes.push_back(n1 - 1);
                        normalIndexes.push_back(n2 - 1);
                        normalIndexes.push_back(n3 - 1);
                        normalIndexes.push_back(n1 - 1);
                        normalIndexes.push_back(n3 - 1);
                        normalIndexes.push_back(n4 - 1);

                        textureIndexes.push_back(t1 - 1);
                        textureIndexes.push_back(t2 - 1);
                        textureIndexes.push_back(t3 - 1);
                        textureIndexes.push_back(t1 - 1);
                        textureIndexes.push_back(t3 - 1);
                        textureIndexes.push_back(t4 - 1);
                    }
                }
                else if (spaces == 3) {
                    if (slashes == 6) {
                        int f1, t1, n1;
                        int f2, t2, n2;
                        int f3, t3, n3;
                        const char* chars = line.c_str();
                        std::sscanf(chars, "f %i/%i/%i %i/%i/%i %i/%i/%i",
                                    &f1, &t1, &n1, &f2, &t2, &n2, &f3, &t3, &n3);
                        faceIndexes.push_back(f1 - 1);
                        faceIndexes.push_back(f2 - 1);
                        faceIndexes.push_back(f3 - 1);

                        normalIndexes.push_back(n1 - 1);
                        normalIndexes.push_back(n2 - 1);
                        normalIndexes.push_back(n3 - 1);

                        textureIndexes.push_back(t1 - 1);
                        textureIndexes.push_back(t2 - 1);
                        textureIndexes.push_back(t3 - 1);
                    } else if (slashes == 3) {
                        int f1, t1;
                        int f2, t2;
                        int f3, t3;
                        const char* chars = line.c_str();
                        std::sscanf(chars, "f %i/%i %i/%i %i/%i",
                                    &f1, &t1, &f2, &t2, &f3, &t3);
                        faceIndexes.push_back(f1 - 1);
                        faceIndexes.push_back(f2 - 1);
                        faceIndexes.push_back(f3 - 1);

                        textureIndexes.push_back(t1 - 1);
                        textureIndexes.push_back(t2 - 1);
                        textureIndexes.push_back(t3 - 1);
                    }
                }
            }
        }
        int buffer_size = (int)faceIndexes.size() / 3;
        Triangle** buffer = new Triangle* [buffer_size];
        if (normalCoords.size() > 0) {
            for (int i = 0; i < faceIndexes.size(); i += 3)
            {
                Vector4* v1 = faceCoords[faceIndexes[i + 0]];
                Vector4* v2 = faceCoords[faceIndexes[i + 1]];
                Vector4* v3 = faceCoords[faceIndexes[i + 2]];

                Vector4* n1 = normalCoords[normalIndexes[i + 0]];
                Vector4* n2 = normalCoords[normalIndexes[i + 1]];
                Vector4* n3 = normalCoords[normalIndexes[i + 2]];

                Vector2* uv1 = textureCoords[textureIndexes[i + 0]];
                Vector2* uv2 = textureCoords[textureIndexes[i + 1]];
                Vector2* uv3 = textureCoords[textureIndexes[i + 2]];

                buffer[i / 3] = new Triangle(v1, v2, v3, n1, n2, n3, uv1, uv2, uv3);
            }
        } else {
            for (int i = 0; i < faceIndexes.size(); i += 3)
            {
                Vector4* v1 = faceCoords[faceIndexes[i + 0]];
                Vector4* v2 = faceCoords[faceIndexes[i + 1]];
                Vector4* v3 = faceCoords[faceIndexes[i + 2]];

                Vector4 *n1 = nullptr, *n2 = nullptr, *n3 = nullptr;

                Vector2* uv1 = textureCoords[textureIndexes[i + 0]];
                Vector2* uv2 = textureCoords[textureIndexes[i + 1]];
                Vector2* uv3 = textureCoords[textureIndexes[i + 2]];

                buffer[i / 3] = new Triangle(v1, v2, v3, n1, n2, n3, uv1, uv2, uv3);
            }
        }
        return *new Object3D(buffer, buffer_size);
    }
};

#endif //OBJECT3D
