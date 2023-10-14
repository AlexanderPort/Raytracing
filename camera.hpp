
#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <SDL.h>
#include "linalg.hpp"


class Camera {
public:
    linalg::vector4 position = {10, 10, 10, 1};
    linalg::vector4 rotation = {0, 0, 0, 1};
    linalg::vector4 forward = {0, 0, 1, 1};
    linalg::vector4 right = {1, 0, 0, 1};
    linalg::vector4 up = {0, 1, 0, 1};
    float near_plane = 1000;
    float far_plane = 10000;
    float h_fov = 0.5f;
    float v_fov = 1;
    float width{}, height{};
    float h_width{}, h_height{};
    float moving_speed = 3.0f;
    float rotation_speed = 10;
    int mouse_x, mouse_y;
    float factor_x, factor_y;
    bool MOUSE_BUTTON_DOWN = false;

    Camera() = default;
    Camera(int width, int height) {
        this->width = (float)width;
        this->height = (float)height;
        this->h_width = (float)width / 2;
        this->h_height = (float)height / 2;
        SDL_GetMouseState(&mouse_x, &mouse_y);
        this->v_fov = this->height / this->width * h_fov;
        this->factor_x = std::tan(this->h_fov / 2);
        this->factor_y = std::tan(this->v_fov / 2);
    }
    linalg::matrix4x4 translate_matrix() const {
        float x = -position.x;
        float y = -position.y;
        float z = -position.z;
        return {
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 1, 0, 1
        };
    }
    linalg::matrix4x4 rotate_matrix() const {
        float fx = forward.x;
        float fy = forward.y;
        float fz = forward.z;
        float rx = right.x;
        float ry = right.y;
        float rz = right.z;
        float ux = up.x;
        float uy = up.y;
        float uz = up.z;
        return {
            rx, ry, rz, 0,
            ux, uy, uz, 0,
            fx, fy, fz, 0,
            0, 0, 0, 1
        };
    }
    linalg::matrix4x4 view_matrix() const {
        return linalg::dot(rotate_matrix(), translate_matrix());
    }
    linalg::matrix4x4 projection_matrix() const {
        float near = near_plane;
        float far = far_plane;
        float right = std::tan(h_fov / 2);
        float top = std::tan(v_fov / 2);
        float left = -right, bottom = -top;

        float m00 = 2 / (right - left);
        float m11 = 2 / (top - bottom);
        float m22 = (far + near) / (far - near);
        float m23 = -2 * near * far / (far - near);

        return {
            m00, 0, 0, 0,
            0, m11, 0, 0,
            0, 0, m22, m23,
            0, 0, 1, 0
        };
    }
    linalg::matrix4x4 screen_matrix() const {
        return {
            h_width, 0, 0, h_width,
            0, -h_height, 0, h_height,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
    }
    bool control(SDL_Event& event) {
        if (event.type == SDL_KEYDOWN) {
            if (event.key.keysym.sym == SDLK_a) {
                position -= right * moving_speed;
                return true;
            }
            if (event.key.keysym.sym == SDLK_d) {
                position += right * moving_speed;
                return true;
            }
            if (event.key.keysym.sym == SDLK_w) {
                position += forward * moving_speed;
                return true;
            }
            if (event.key.keysym.sym == SDLK_s) {
                position -= forward * moving_speed;
                return true;
            }
            if (event.key.keysym.sym == SDLK_q) {
                position += up * moving_speed;
                return true;
            }
            if (event.key.keysym.sym == SDLK_e) {
                position -= up * moving_speed;
                return true;
            }
            /*
            if (event.key.keysym.sym == SDLK_UP) {
                rotate_y(-rotation_speed);
            }
            if (event.key.keysym.sym == SDLK_DOWN) {
                rotate_y(rotation_speed);
            }
            if (event.key.keysym.sym == SDLK_RIGHT) {
                rotate_x(rotation_speed);
            }
            if (event.key.keysym.sym == SDLK_LEFT) {
                rotate_x(-rotation_speed);
            }
            */
        } else if (event.type == SDL_MOUSEBUTTONUP) {
            MOUSE_BUTTON_DOWN = false;
        } else if (event.type == SDL_MOUSEBUTTONDOWN) {
            MOUSE_BUTTON_DOWN = true;
            mouse_x = event.motion.x; 
            mouse_y = event.motion.y;
        } else if (event.type == SDL_MOUSEMOTION && MOUSE_BUTTON_DOWN) {
            float x = (mouse_x - event.motion.x) / width * 60;
            float y = (mouse_y - event.motion.y) / height * 60;
            mouse_x = event.motion.x; rotation.x -= x;
            mouse_y = event.motion.y; rotation.y -= y;
            rotate_y(-y); rotate_x(-x);
            return true;
        }
        return false;
    }
    void rotate_x(float degrees) {
        linalg::matrix4x4 rotate = linalg::rotate_x(degrees);
        forward = linalg::dot(rotate, forward);
        right = linalg::dot(rotate, right);
        up = linalg::dot(rotate, up);
    }
    void rotate_y(float degrees) {
        linalg::matrix4x4 rotate = linalg::rotate_y(degrees);
        forward = linalg::dot(rotate, forward);
        right = linalg::dot(rotate, right);
        up = linalg::dot(rotate, up);
    }
};
#endif //CAMERA_HPP