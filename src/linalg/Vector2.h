//
// Created by alexander on 24.02.2021.
//

#ifndef VECTOR2
#define VECTOR2

class Vector2 {
public:
    float x, y;
    Vector2() {
        this->x = 0;
        this->y = 0;
    }
    Vector2(float x, float y) {
        this->x = x;
        this->y = y;
    }
    explicit Vector2(const float* xy) {
        this->x = xy[0];
        this->y = xy[1];
    }
    float length() {
        return std::sqrt(x * x + y * y);
    }
    friend Vector2 operator+(const Vector2& vector1, const Vector2& vector2) {
        return { vector1.x + vector2.x, vector1.y + vector2.y };
    }
    friend Vector2 operator-(const Vector2& vector1, const Vector2& vector2) {
        return { vector1.x - vector2.x, vector1.y - vector2.y };
    }
    friend Vector2 operator*(const Vector2& vector1, const Vector2& vector2) {
        return { vector1.x * vector2.x, vector1.y * vector2.y };
    }
    friend Vector2 operator/(const Vector2& vector1, const Vector2& vector2) {
        return { vector1.x / vector2.x, vector1.y / vector2.y };
    }
    friend Vector2 operator+(const Vector2& vector, float scalar) {
        return { vector.x + scalar, vector.y + scalar };
    }
    friend Vector2 operator-(const Vector2& vector, float scalar) {
        return { vector.x - scalar, vector.y - scalar };
    }
    friend Vector2 operator*(const Vector2& vector, float scalar) {
        return { vector.x * scalar, vector.y * scalar };
    }
    friend Vector2 operator/(const Vector2& vector, float scalar) {
        return { vector.x / scalar, vector.y / scalar };
    }
    friend Vector2 operator+(float scalar, const Vector2& vector) {
        return { scalar + vector.x, scalar + vector.y };
    }
    friend Vector2 operator-(float scalar, const Vector2& vector) {
        return { scalar - vector.x, scalar - vector.y };
    }
    friend Vector2 operator*(float scalar, const Vector2& vector) {
        return { scalar * vector.x, scalar * vector.y };
    }
    friend Vector2 operator/(float scalar, const Vector2& vector) {
        return { scalar / vector.x, scalar / vector.y };
    }

};

#endif //VECTOR2
