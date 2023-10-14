#ifndef LINALG_HPP
#define LINALG_HPP

#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

#include <cmath>

namespace linalg {

    static float degree_to_radian = 3.141592653589793238463f / 180.0f;
    static float radian_to_degree = 180.0f / 3.141592653589793238463f;

    typedef struct vector2 {
        float x, y;
    } vector2;

    typedef struct vector3 {
        float x, y, z;
        __host__ __device__ vector3() {
            this->x = 0; this->y = 0; this->z = 0;
        }
        __host__ __device__ vector3(float x, float y, float z) {
            this->x = x; this->y = y; this->z = z;
        }
        __host__ __device__ friend void operator+=(vector3& f1, float f2) {
            f1.x += f2; f1.y += f2; f1.z += f2;
        }
        __host__ __device__ friend void operator-=(vector3& f1, float f2) {
            f1.x -= f2; f1.y -= f2; f1.z -= f2;
        }
        __host__ __device__ friend void operator*=(vector3& f1, float f2) {
            f1.x *= f2; f1.y *= f2; f1.z *= f2;
        }
        __host__ __device__ friend void operator/=(vector3& f1, float f2) {
            f1.x /= f2; f1.y /= f2; f1.z /= f2;
        }
        __host__ __device__ friend void operator+=(vector3& f1, const vector3& f2) {
            f1.x += f2.x; f1.y += f2.y; f1.z += f2.z;
        }
        __host__ __device__ friend void operator-=(vector3& f1, const vector3& f2) {
            f1.x -= f2.x; f1.y -= f2.y; f1.z -= f2.z;
        }
        __host__ __device__ friend void operator*=(vector3& f1, const vector3& f2) {
            f1.x *= f2.x; f1.y *= f2.y; f1.z *= f2.z;
        }
        __host__ __device__ friend void operator/=(vector3& f1, const vector3& f2) {
            f1.x /= f2.x; f1.y /= f2.y; f1.z /= f2.z;
        }
        __host__ __device__ friend vector3 operator+(const vector3& f1, float f2) {
            return {f1.x + f2, f1.y + f2, f1.z + f2};
        }
        __host__ __device__ friend vector3 operator-(const vector3& f1, float f2) {
            return {f1.x - f2, f1.y - f2, f1.z - f2};
        }
        __host__ __device__ friend vector3 operator*(const vector3& f1, float f2) {
            return {f1.x * f2, f1.y * f2, f1.z * f2};
        }
        __host__ __device__ friend vector3 operator/(const vector3& f1, float f2) {
            return {f1.x / f2, f1.y / f2, f1.z / f2};
        }
        __host__ __device__ friend vector3 operator+(float f1, const vector3& f2) {
            return {f1 + f2.x, f1 + f2.y, f1 + f2.z};
        }
        __host__ __device__ friend vector3 operator-(float f1, const vector3& f2) {
            return {f1 - f2.x, f1 - f2.y, f1 - f2.z};
        }
        __host__ __device__ friend vector3 operator*(float f1, const vector3& f2) {
            return {f1 * f2.x, f1 * f2.y, f1 * f2.z};
        }
        __host__ __device__ friend vector3 operator/(float f1, const vector3& f2) {
            return {f1 / f2.x, f1 / f2.y, f1 / f2.z};
        }
        __host__ __device__ friend vector3 operator+(const vector3& f1, const vector3& f2) {
            return {f1.x + f2.x, f1.y + f2.y, f1.z + f2.z};
        }
        __host__ __device__ friend vector3 operator-(const vector3& f1, const vector3& f2) {
            return {f1.x - f2.x, f1.y - f2.y, f1.z - f2.z};
        }
        __host__ __device__ friend vector3 operator*(const vector3& f1, const vector3& f2) {
            return {f1.x * f2.x, f1.y * f2.y, f1.z * f2.z};
        }
        __host__ __device__ friend vector3 operator/(const vector3& f1, const vector3& f2) {
            return {f1.x / f2.x, f1.y / f2.y, f1.z / f2.z};
        }
        __host__ __device__ friend vector3 operator-(const vector3& f) {
            return {-f.x, -f.y, -f.y};
        }
    } vector3;

    typedef struct vector4 {
        float x, y, z, w;
        __host__ __device__ vector4(float x, float y, float z, float w) {
            this->x = x; this->y = y; this->z = z; this->w = w;
        }
        __host__ __device__ friend void operator+=(vector4& f1, float f2) {
            f1.x += f2; f1.y += f2; f1.z += f2; f1.w += f2;
        }
        __host__ __device__ friend void operator-=(vector4& f1, float f2) {
            f1.x -= f2; f1.y -= f2; f1.z -= f2; f1.w -= f2;
        }
        __host__ __device__ friend void operator*=(vector4& f1, float f2) {
            f1.x *= f2; f1.y *= f2; f1.z *= f2; f1.w *= f2;
        }
        __host__ __device__ friend void operator/=(vector4& f1, float f2) {
            f1.x /= f2; f1.y /= f2; f1.z /= f2; f1.w /= f2;
        }
        __host__ __device__ friend void operator+=(vector4& f1, const vector4& f2) {
            f1.x += f2.x; f1.y += f2.y; f1.z += f2.z; f1.w += f2.w;
        }
        __host__ __device__ friend void operator-=(vector4& f1, const vector4& f2) {
            f1.x -= f2.x; f1.y -= f2.y; f1.z -= f2.z; f1.w -= f2.w;
        }
        __host__ __device__ friend void operator*=(vector4& f1, const vector4& f2) {
            f1.x *= f2.x; f1.y *= f2.y; f1.z *= f2.z; f1.w *= f2.w;
        }
        __host__ __device__ friend void operator/=(vector4& f1, const vector4& f2) {
            f1.x /= f2.x; f1.y /= f2.y; f1.z /= f2.z; f1.w /= f2.w;
        }
        __host__ __device__ friend vector4 operator+(const vector4& f1, float f2) {
            return {f1.x + f2, f1.y + f2, f1.z + f2, f1.w + f2};
        }
        __host__ __device__ friend vector4 operator-(const vector4& f1, float f2) {
            return {f1.x - f2, f1.y - f2, f1.z - f2, f1.w - f2};
        }
        __host__ __device__ friend vector4 operator*(const vector4& f1, float f2) {
            return {f1.x * f2, f1.y * f2, f1.z * f2, f1.w * f2};
        }
        __host__ __device__ friend vector4 operator/(const vector4& f1, float f2) {
            return {f1.x / f2, f1.y / f2, f1.z / f2, f1.w / f2};
        }
        __host__ __device__ friend vector4 operator+(float f1, const vector4& f2) {
            return {f1 + f2.x, f1 + f2.y, f1 + f2.z, f1 + f2.w};
        }
        __host__ __device__ friend vector4 operator-(float f1, const vector4& f2) {
            return {f1 - f2.x, f1 - f2.y, f1 - f2.z, f1 - f2.w};
        }
        __host__ __device__ friend vector4 operator*(float f1, const vector4& f2) {
            return {f1 * f2.x, f1 * f2.y, f1 * f2.z, f1 * f2.w};
        }
        __host__ __device__ friend vector4 operator/(float f1, const vector4& f2) {
            return {f1 / f2.x, f1 / f2.y, f1 / f2.z, f1 / f2.w};
        }
        __host__ __device__ friend vector4 operator+(const vector4& f1, const vector4& f2) {
            return {f1.x + f2.x, f1.y + f2.y, f1.z + f2.z, f1.w + f2.w};
        }
        __host__ __device__ friend vector4 operator-(const vector4& f1, const vector4& f2) {
            return {f1.x - f2.x, f1.y - f2.y, f1.z - f2.z, f1.w - f2.w};
        }
        __host__ __device__ friend vector4 operator*(const vector4& f1, const vector4& f2) {
            return {f1.x * f2.x, f1.y * f2.y, f1.z * f2.z, f1.w * f2.w};
        }
        __host__ __device__ friend vector4 operator/(const vector4& f1, const vector4& f2) {
            return {f1.x / f2.x, f1.y / f2.y, f1.z / f2.z, f1.w / f2.w};
        }
    } vector4;

    typedef struct matrix4x4 {
        float f00, f01, f02, f03;
        float f10, f11, f12, f13;
        float f20, f21, f22, f23;
        float f30, f31, f32, f33;
    } matrix4x4;

    static __host__ __device__ vector3 xyz(const vector4& f) { return {f.x, f.y, f.z}; }

    static __host__ __device__ float dot(const vector3& f1, const vector3& f2) {
        return f1.x * f2.x + f1.y * f2.y  + f1.z * f2.z;
    }
    static vector4 dot(const matrix4x4& f1, const vector4& f2) {
        return {
            f1.f00 * f2.x + f1.f01 * f2.y + f1.f02 * f2.z + f1.f03 * f2.w,
            f1.f10 * f2.x + f1.f11 * f2.y + f1.f12 * f2.z + f1.f13 * f2.w,
            f1.f20 * f2.x + f1.f21 * f2.y + f1.f22 * f2.z + f1.f23 * f2.w,
            f1.f30 * f2.x + f1.f31 * f2.y + f1.f32 * f2.z + f1.f33 * f2.w
        };
    }

    static __host__ __device__ linalg::vector3 cross(const linalg::vector3& u, const linalg::vector3& v) {
        return {u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
    }

    static matrix4x4 dot(const matrix4x4& f1, const matrix4x4& f2) {
        return {
            f1.f00 * f2.f00 + f1.f01 * f2.f10 + f1.f02 * f2.f20 + f1.f03 * f2.f30,
            f1.f00 * f2.f01 + f1.f01 * f2.f11 + f1.f02 * f2.f21 + f1.f03 * f2.f31,
            f1.f00 * f2.f02 + f1.f01 * f2.f12 + f1.f02 * f2.f22 + f1.f03 * f2.f32,
            f1.f00 * f2.f03 + f1.f01 * f2.f13 + f1.f02 * f2.f23 + f1.f03 * f2.f33,
            f1.f10 * f2.f00 + f1.f11 * f2.f10 + f1.f12 * f2.f20 + f1.f13 * f2.f30,
            f1.f10 * f2.f01 + f1.f11 * f2.f11 + f1.f12 * f2.f21 + f1.f13 * f2.f31,
            f1.f10 * f2.f02 + f1.f11 * f2.f12 + f1.f12 * f2.f22 + f1.f13 * f2.f32,
            f1.f10 * f2.f03 + f1.f11 * f2.f13 + f1.f12 * f2.f23 + f1.f13 * f2.f33,
            f1.f20 * f2.f00 + f1.f21 * f2.f10 + f1.f22 * f2.f20 + f1.f23 * f2.f30,
            f1.f20 * f2.f01 + f1.f21 * f2.f11 + f1.f22 * f2.f21 + f1.f23 * f2.f31,
            f1.f20 * f2.f02 + f1.f21 * f2.f12 + f1.f22 * f2.f22 + f1.f23 * f2.f32,
            f1.f20 * f2.f03 + f1.f21 * f2.f13 + f1.f22 * f2.f23 + f1.f23 * f2.f33,
            f1.f30 * f2.f00 + f1.f31 * f2.f10 + f1.f32 * f2.f20 + f1.f33 * f2.f30,
            f1.f30 * f2.f01 + f1.f31 * f2.f11 + f1.f32 * f2.f21 + f1.f33 * f2.f31,
            f1.f30 * f2.f02 + f1.f31 * f2.f12 + f1.f32 * f2.f22 + f1.f33 * f2.f32,
            f1.f30 * f2.f03 + f1.f31 * f2.f13 + f1.f32 * f2.f23 + f1.f33 * f2.f33
        };
    }
    static matrix4x4 scale(float x, float y, float z) {
        return {
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1
        };
    }
    static matrix4x4 translate(float x, float y, float z) {
        return {
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1
        };
    }
    static matrix4x4 rotate_y(float degrees) {
        float radians = degree_to_radian * degrees;
        float sin = std::sin(radians);
        float cos = std::cos(radians);
        return {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, +cos, -sin, 0.0f,
            0.0f, +sin, +cos, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
    }
    static matrix4x4 rotate_x(float degrees) {
        float radians = degree_to_radian * degrees;
        float sin = std::sin(radians);
        float cos = std::cos(radians);
        return {
            +cos, 0.0f, +sin, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            -sin, 0.0f, +cos, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
    }
    static matrix4x4 rotate_z(float degrees) {
        float radians = degree_to_radian * degrees;
        float sin = std::sin(radians);
        float cos = std::cos(radians);
        return {
            +cos, -sin, 0.0f, 0.0f,
            +sin, +cos, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
    }
    
    static matrix4x4 rotate(float x, float y, float z) {
        return dot(rotate_z(z), dot(rotate_y(y), rotate_x(x)));
    }

    static __host__ __device__ float length(const vector3& f) {
        return std::sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
    }

    static __host__ __device__ vector3 normalize(const vector3& f) {
        float length = std::sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
        return {f.x / length, f.y / length, f.z / length};
    }
    static bool near_zero(const vector3& vector) {
        const float s = 1e-6;
        return (fabs(vector.x) < s) && (fabs(vector.y) < s) && (fabs(vector.z) < s);
    }
    static __device__ linalg::vector3 random_vector(curandState* random) {
        return {curand_uniform(random), curand_uniform(random), curand_uniform(random)};
    }
    static __host__ __device__ float length_squared(const linalg::vector3& v) {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }
}

#endif //LINALG_HPP