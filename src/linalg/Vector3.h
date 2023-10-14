//
// Created by alexander on 02.03.2021.
//

#ifndef VECTOR3
#define VECTOR3
#include "Vector4.h"

class Vector3 {
public:
    float x, y, z;
    Vector3 () {
        this->x = 0;
        this->y = 0;
        this->z = 0;
    }
    Vector3(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }
    Vector3(Vector4 vector) {
        this->x = vector.x;
        this->y = vector.y;
        this->z = vector.z;
    }

    Vector3 operator-() { return Vector3(-x, -y, -z); }

    friend Vector3 operator% (const Vector3& vector1, const Vector3& vector2) {
        return Vector3(vector1.y * vector2.z - vector1.z * vector2.y,
                       vector1.z * vector2.x - vector1.x * vector2.z,
                       vector1.x * vector2.y - vector1.y * vector2.x);
    }
    friend Vector3 operator% (const Vector3& vector1, const Vector4& vector2) {
        return Vector3(vector1.y * vector2.z - vector1.z * vector2.y,
                       vector1.z * vector2.x - vector1.x * vector2.z,
                       vector1.x * vector2.y - vector1.y * vector2.x);
    }
    friend Vector3 operator% (const Vector4& vector1, const Vector3& vector2) {
        return Vector3(vector1.y * vector2.z - vector1.z * vector2.y,
                       vector1.z * vector2.x - vector1.x * vector2.z,
                       vector1.x * vector2.y - vector1.y * vector2.x);
    }
    float dot(const Vector3& vector) {
        return x * vector.x + y * vector.y + z * vector.z;
    }
    float dot(const Vector4& vector) {
        return x * vector.x + y * vector.y + z * vector.z;
    }
    static float dot(const Vector3& vector1, const Vector3& vector2) {
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z;
    }
    friend Vector3 operator-(Vector4& vector1, Vector3& vector2) {
        return {
            vector1.x - vector2.x,
            vector1.y - vector2.y,
            vector1.z - vector2.z,
        };
    }
    Vector3 normalize() {
        float length = sqrt(x * x + y * y + z * z);
        return Vector3(x / length, y / length, z / length);
    }
    inline float length() {
        return std::sqrt(x * x + y * y + z * z);
    }
    inline float squared_length() {
        return x * x + y * y + z * z;
    }
    friend Vector3 operator+(Vector3 vector1, Vector3 vector2) {
        return {
                vector1.x + vector2.x,
                vector1.y + vector2.y,
                vector1.z + vector2.z,
        };
    }
    friend Vector3 operator-(Vector3 vector1, Vector3 vector2) {
        return {
            vector1.x - vector2.x,
            vector1.y - vector2.y,
            vector1.z - vector2.z,
        };
    }
    friend Vector3 operator-(Vector3& vector1, Vector3& vector2) {
        return {
            vector1.x - vector2.x,
            vector1.y - vector2.y,
            vector1.z - vector2.z,
        };
    }
    friend Vector3 operator*(Vector3& vector1, Vector3& vector2) {
        return {
                vector1.x * vector2.x,
                vector1.y * vector2.y,
                vector1.z * vector2.z,
        };
    }
    friend Vector3 operator*(Vector3& vector, float scalar) {
        return {
                vector.x * scalar,
                vector.y * scalar,
                vector.z * scalar,
        };
    }
    friend Vector3 operator*(float scalar, Vector3& vector) {
        return {
                scalar * vector.x,
                scalar * vector.y,
                scalar * vector.z,
        };
    }
    friend Vector3 operator/(Vector3& vector1, Vector3& vector2) {
        return {
                vector1.x / vector2.x,
                vector1.y / vector2.y,
                vector1.z / vector2.z,
        };
    }
    friend Vector3 operator/(Vector3& vector, float scalar) {
        return {
                vector.x / scalar,
                vector.y / scalar,
                vector.z / scalar,
        };
    }
    friend Vector3 operator/(float scalar, Vector3& vector) {
        return {
                scalar / vector.x,
                scalar / vector.y,
                scalar / vector.z,
        };
    }
    friend Vector3 operator/(Vector3 vector1, Vector3 vector2) {
        return {
                vector1.x / vector2.x,
                vector1.y / vector2.y,
                vector1.z / vector2.z,
        };
    }
    friend Vector3 operator/(Vector3 vector, float scalar) {
        return {
                vector.x / scalar,
                vector.y / scalar,
                vector.z / scalar,
        };
    }
    friend Vector3 operator/(float scalar, Vector3 vector) {
        return {
                scalar / vector.x,
                scalar / vector.y,
                scalar / vector.z,
        };
    }
    friend void operator+=(Vector3& vector, float scalar) {
        vector.x += scalar;
        vector.y += scalar;
        vector.z += scalar;
    }
    friend void operator-=(Vector3& vector, float scalar) {
        vector.x -= scalar;
        vector.y -= scalar;
        vector.z -= scalar;
    }
    inline float sum() {
        return x + y + z;
    };
    void print() const {
        std::printf("Vector(%f, %f, %f)\n", x, y, z);
    }

};

#endif //VECTOR3
