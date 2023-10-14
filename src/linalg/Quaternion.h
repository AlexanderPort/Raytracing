//
// Created by alexander on 07.03.2021.
//

#ifndef QUATERNION
#define QUATERNION



class Quaternion {
public:
    float x, y, z, w;
    Quaternion(float x, float y, float z, float w) {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
    friend Quaternion operator*(Quaternion& q1, Quaternion& q2) {
        return {
                q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x,
                -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y,
                q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z,
                -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w
        };
    }
    friend Quaternion operator+(Quaternion& q, float scalar) {
        return {
            q.x, q.y, q.z, q.w + scalar
        };
    }
};

#endif //QUATERNION
