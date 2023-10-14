#ifndef VECTOR4
#define VECTOR4

#include <cmath>
#include <iostream>

class Vector4 {
public:
    float x, y, z, w;
    Vector4() {
        this->x = 0;
        this->y = 0;
        this->z = 0;
        this->w = 0;
    }
    Vector4(float x, float y) {
        this->x = x;
        this->y = y;
        this->z = 1;
        this->w = 1;
    }
    Vector4(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = 1;
    }
    Vector4(float x, float y, float z, float w) {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
    explicit Vector4(const float* xyzw) {
        this->x = xyzw[0];
        this->y = xyzw[1];
        this->z = xyzw[2];
        this->w = xyzw[3];
    }
    explicit Vector4(const float* xyz, float w) {
        this->x = xyz[0];
        this->y = xyz[1];
        this->z = xyz[2];
        this->w = w;
    }

    Vector4 operator-() { return Vector4(-x, -y, -z, -w); }

    friend Vector4 operator+(const Vector4 &vector1, const Vector4 &vector2) {
        return Vector4(vector1.x + vector2.x, vector1.y + vector2.y,
                       vector1.z + vector2.z, vector1.w + vector2.w);
    }
    friend Vector4 operator-(const Vector4 &vector1, const Vector4 &vector2) {
        return Vector4(vector1.x - vector2.x, vector1.y - vector2.y,
                       vector1.z - vector2.z, vector1.w - vector2.w);
    }
    friend Vector4 operator*(const Vector4 &vector1, const Vector4 &vector2) {
        return Vector4(vector1.x * vector2.x, vector1.y * vector2.y,
                       vector1.z * vector2.z, vector1.w * vector2.w);
    }
    friend Vector4 operator/(const Vector4 &vector1, const Vector4 &vector2) {
        return Vector4(vector1.x / vector2.x, vector1.y / vector2.y,
                       vector1.z / vector2.z, vector1.w / vector2.w);
    }
    friend Vector4 operator+(const Vector4 &vector, const float scalar) {
        return Vector4(vector.x + scalar, vector.y + scalar,
                       vector.z + scalar, vector.w + scalar);
    }
    friend Vector4 operator-(const Vector4 &vector, const float scalar) {
        return Vector4(vector.x - scalar, vector.y - scalar,
                       vector.z - scalar, vector.w - scalar);
    }
    friend Vector4 operator*(const Vector4 &vector, const float scalar) {
        return Vector4(vector.x * scalar, vector.y * scalar,
                       vector.z * scalar, vector.w * scalar);
    }
    friend Vector4 operator/(const Vector4 &vector, const float scalar) {
        return Vector4(vector.x / scalar, vector.y / scalar,
                       vector.z / scalar, vector.w / scalar);
    }
    friend Vector4 operator+(const float scalar, const Vector4 &vector) {
        return Vector4(scalar + vector.x, scalar + vector.y,
                       scalar + vector.z, scalar + vector.w);
    }
    friend Vector4 operator-(const float scalar, const Vector4 &vector) {
        return Vector4(scalar - vector.x, scalar - vector.y,
                       scalar - vector.z, scalar - vector.w);
    }
    friend Vector4 operator*(const float scalar, const Vector4 &vector) {
        return Vector4(scalar * vector.x, scalar * vector.y,
                       scalar * vector.z, scalar * vector.w);
    }
    friend Vector4 operator/(const float scalar, const Vector4 &vector) {
        return Vector4(scalar / vector.x, scalar / vector.y,
                       scalar / vector.z, scalar / vector.w);
    };
    friend void operator+=(Vector4 &vector1, const Vector4 &vector2) {
        vector1.x += vector2.x;
        vector1.y += vector2.y;
        vector1.z += vector2.z;
        vector1.w += vector2.w;
    }
    friend void operator-=(Vector4 &vector1, const Vector4 &vector2) {
        vector1.x -= vector2.x;
        vector1.y -= vector2.y;
        vector1.z -= vector2.z;
        vector1.w -= vector2.w;
    }
    friend void operator*=(Vector4 &vector1, const Vector4 &vector2) {
        vector1.x *= vector2.x;
        vector1.y *= vector2.y;
        vector1.z *= vector2.z;
        vector1.w *= vector2.w;
    }
    friend void operator/=(Vector4 &vector1, const Vector4 &vector2) {
        vector1.x /= vector2.x;
        vector1.y /= vector2.y;
        vector1.z /= vector2.z;
        vector1.w /= vector2.w;
    }
    friend void operator+=(Vector4 &vector1, const float scalar) {
        vector1.x += scalar;
        vector1.y += scalar;
        vector1.z += scalar;
        vector1.w += scalar;
    }
    friend void operator-=(Vector4 &vector1, const float scalar) {
        vector1.x -= scalar;
        vector1.y -= scalar;
        vector1.z -= scalar;
        vector1.w -= scalar;
    }
    friend void operator*=(Vector4 &vector1, const float scalar) {
        vector1.x *= scalar;
        vector1.y *= scalar;
        vector1.z *= scalar;
        vector1.w *= scalar;
    }
    friend void operator/=(Vector4 &vector1, const float scalar) {
        vector1.x /= scalar;
        vector1.y /= scalar;
        vector1.z /= scalar;
        vector1.w /= scalar;
    }
    friend Vector4 operator%(const Vector4& vector1, const Vector4& vector2) {
        return Vector4(vector1.y * vector2.z - vector1.z * vector2.y,
                       vector1.z * vector2.x - vector1.x * vector2.z,
                       vector1.x * vector2.y - vector1.y * vector2.x);
    }
    float length(bool mode = true) const {
        return (float)std::sqrt(x * x + y * y + z * z + w * w);
    }
    Vector4 normalize() const {
        float length = std::sqrt(x * x + y * y + z * z);
        return Vector4(x / length, y / length, z / length, 1);
    }
    void print() const {
        std::printf("Vector(%f, %f, %f, %f)\n", x, y, z, w);
    }
    static float edge(const Vector4& v1, const Vector4& v2, const Vector4& v3) {
        return (v3.x - v1.x) * (v2.y - v1.y) - (v3.y - v1.y) * (v2.x - v1.x);
    }

};


#endif //VECTOR4
