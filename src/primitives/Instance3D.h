//
// Created by alexander on 23.02.2021.
//

#ifndef INSTANCE3D
#define INSTANCE3D
#include "Object3D.h"
#include "../rendering/Renderer.h"
#include "../linalg/Vector2.h"
#include "../linalg/Vector3.h"
#include "../linalg/Vector4.h"
#include "../linalg/Matrix4x4.h"

#include <algorithm>

class Instance3D {
public:
    Object3D* object3D;
    bool phong_model = false;
    float specular_power = 100.0f;
    Vector4 scale = Vector4(1, 1, 1, 1);
    Vector4 position = Vector4(0, 0, 0, 1);
    Vector4 rotation = Vector4(0, 0, 0, 1);
    Vector4 ambient = Vector4(1.0f, 1.0f, 1.0f);
    Vector4 diffuse = Vector4(1.0f, 1.0f, 1.0f);
    Vector4 specular = Vector4(1.0f, 1.0f, 1.0f);
    Instance3D() {
        object3D = nullptr;
    }
    Instance3D(Object3D* object3D) {
        this->object3D = object3D;
    }
    Instance3D(Object3D* object3D, Vector4 position, Vector4 rotation, Vector4 scale) {
        this->object3D = object3D;
        this->position = position;
        this->rotation = rotation;
        this->scale = scale;
    }
    inline Matrix4x4 model_matrix() const {
        return translate_matrix() * rotate_matrix() * scale_matrix();
    }
    inline Matrix4x4 rotate_matrix() const {
        Matrix4x4 rotate_x = Matrix4x4::rotate_x(rotation.x);
        Matrix4x4 rotate_y = Matrix4x4::rotate_y(rotation.y);
        Matrix4x4 rotate_z = Matrix4x4::rotate_z(rotation.z);
        return rotate_z * rotate_y * rotate_x;
    }
    inline Matrix4x4 translate_matrix() const {
        return Matrix4x4::translate(position.x, position.y, position.z);
    }
    inline Matrix4x4 scale_matrix() const {
        return Matrix4x4::scale(scale.x, scale.y, scale.z);
    }
    void update() {
        
    }
    void render(Renderer* renderer) {
        float specular_power = 500.0f;
        Matrix4x4 mModelView = renderer->mView * model_matrix();
        Vector3 view = renderer->camera.position;
    #pragma omp parallel for
        for (int i = 0; i < object3D->buffer_size; i++) {
            Triangle* triangle = object3D->buffer[i];
            Vector4 v1 = mModelView * *triangle->v1;
            Vector4 v2 = mModelView * *triangle->v2;
            Vector4 v3 = mModelView * *triangle->v3;

            if (phong_model) { 
                Vector3 N1 = mModelView * *triangle->n1;
                Vector3 N2 = mModelView * *triangle->n2;
                Vector3 N3 = mModelView * *triangle->n3;

                Vector3 L1 = -v1; Vector3 L2 = -v2; Vector3 L3 = -v3;
                Vector3 V1 = -v1; Vector3 V2 = -v2; Vector3 V3 = -v3;

                N1 = N1.normalize(); N2 = N2.normalize(); N3 = N3.normalize();
                L1 = L1.normalize(); L2 = L2.normalize(); L3 = L3.normalize();
                V1 = V1.normalize(); V2 = V2.normalize(); V3 = V3.normalize();

                Vector3 R1 = -L1 - 2 * Vector3::dot(N1, -L1) * N1;
                Vector3 R2 = -L2 - 2 * Vector3::dot(N2, -L2) * N2;
                Vector3 R3 = -L3 - 2 * Vector3::dot(N3, -L3) * N3;

                Vector3 diffuse1 = std::max(Vector3::dot(N1, L1), 0.0f) * diffuse;
                Vector3 diffuse2 = std::max(Vector3::dot(N2, L2), 0.0f) * diffuse;
                Vector3 diffuse3 = std::max(Vector3::dot(N3, L3), 0.0f) * diffuse;

                Vector3 specular1 = std::pow(std::max(Vector3::dot(R1, V1), 0.0f), specular_power) * specular;
                Vector3 specular2 = std::pow(std::max(Vector3::dot(R2, V2), 0.0f), specular_power) * specular;
                Vector3 specular3 = std::pow(std::max(Vector3::dot(R3, V3), 0.0f), specular_power) * specular;

                if (v1.z < 0 && v2.z < 0 && v3.z < 0) continue;
                v1 *= renderer->mScreenProjection;
                v2 *= renderer->mScreenProjection;
                v3 *= renderer->mScreenProjection;
                Vector2 uv1 = *triangle->uv1;
                Vector2 uv2 = *triangle->uv2;
                Vector2 uv3 = *triangle->uv3;
                v1.x /= v1.w; v1.y /= v1.w;
                v2.x /= v2.w; v2.y /= v2.w;
                v3.x /= v3.w; v3.y /= v3.w;            
                renderer->rasterizer.rasterize(
                    v1, v2, v3, uv1, uv2, uv3,
                    diffuse1, diffuse2, diffuse3,
                    specular1, specular2, specular3, 
                    object3D->texture);
            } else {
                if (v1.z < 0 && v2.z < 0 && v3.z < 0) continue;
                v1 *= renderer->mScreenProjection;
                v2 *= renderer->mScreenProjection;
                v3 *= renderer->mScreenProjection;
                Vector2 uv1 = *triangle->uv1;
                Vector2 uv2 = *triangle->uv2;
                Vector2 uv3 = *triangle->uv3;
                v1.x /= v1.w; v1.y /= v1.w;
                v2.x /= v2.w; v2.y /= v2.w;
                v3.x /= v3.w; v3.y /= v3.w;            
                renderer->rasterizer.rasterize(
                    v1, v2, v3, uv1, uv2, uv3,
                    object3D->texture);
            }
        }

    }

    void rotate_around(Vector4 point, Vector4 angles) {
        position = Matrix4x4::rotate(angles.x, angles.y, angles.z) * (position - point) + point;
    }

};

#endif //INSTANCE3D
