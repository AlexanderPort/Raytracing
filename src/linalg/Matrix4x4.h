#ifndef MATRIX4X4
#define MATRIX4X4

#include <cmath>
#include <iostream>
#include "Vector4.h"
#include <vector>



static float degree_to_radian = 3.141592653589793238463f / 180.0f;
static float radian_to_degree = 180.0f / 3.141592653589793238463f;


class Matrix4x4 {
public:
    float m00, m01, m02, m03;
    float m10, m11, m12, m13;
    float m20, m21, m22, m23;
    float m30, m31, m32, m33;

    explicit Matrix4x4() {
        this->m00 = 0.0f; this->m01 = 0.0f; this->m02 = 0.0f; this->m03 = 0.0f;
        this->m10 = 0.0f; this->m11 = 0.0f; this->m12 = 0.0f; this->m13 = 0.0f;
        this->m20 = 0.0f; this->m21 = 0.0f; this->m22 = 0.0f; this->m23 = 0.0f;
        this->m30 = 0.0f; this->m31 = 0.0f; this->m32 = 0.0f; this->m33 = 0.0f;
    }

    explicit Matrix4x4(
            float m00, float m01, float m02, float m03,
            float m10, float m11, float m12, float m13,
            float m20, float m21, float m22, float m23,
            float m30, float m31, float m32, float m33) {
        this->m00 = m00; this->m01 = m01; this->m02 = m02; this->m03 = m03;
        this->m10 = m10; this->m11 = m11; this->m12 = m12; this->m13 = m13;
        this->m20 = m20; this->m21 = m21; this->m22 = m22; this->m23 = m23;
        this->m30 = m30; this->m31 = m31; this->m32 = m32; this->m33 = m33;
    }

    static Matrix4x4 zeros() {
        return Matrix4x4(
                0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f);
    }

    static Matrix4x4 identity() {
        return Matrix4x4(
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f);
    }
    static Matrix4x4 scale(float x, float y, float z) {
        return Matrix4x4(
                x, 0, 0, 0,
                0, y, 0, 0,
                0, 0, z, 0,
                0, 0, 0, 1
        );
    }
    static Matrix4x4 translate(float x, float y, float z) {
        return Matrix4x4(
                1, 0, 0, x,
                0, 1, 0, y,
                0, 0, 1, z,
                0, 0, 0, 1
        );
    }
    static Matrix4x4 rotate_y(float degrees) {
        float radians = degree_to_radian * degrees;
        float sin = std::sin(radians);
        float cos = std::cos(radians);
        return Matrix4x4(
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, +cos, -sin, 0.0f,
                0.0f, +sin, +cos, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f
        );
    }
    static Matrix4x4 rotate_x(float degrees) {
        float radians = degree_to_radian * degrees;
        float sin = std::sin(radians);
        float cos = std::cos(radians);
        return Matrix4x4(
                +cos, 0.0f, +sin, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                -sin, 0.0f, +cos, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f
        );
    }
    static Matrix4x4 rotate_z(float degrees) {
        float radians = degree_to_radian * degrees;
        float sin = std::sin(radians);
        float cos = std::cos(radians);
        return Matrix4x4(
                +cos, -sin, 0.0f, 0.0f,
                +sin, +cos, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f
        );
    }
    inline static Matrix4x4 rotate(float x, float y, float z) {
        return Matrix4x4::rotate_z(z) * Matrix4x4::rotate_y(y) * Matrix4x4::rotate_x(x);
    }
    friend Matrix4x4 operator*(const Matrix4x4 &matrix1, const Matrix4x4 &matrix2) {
        return Matrix4x4(
                matrix1.m00 * matrix2.m00 + matrix1.m01 * matrix2.m10 + matrix1.m02 * matrix2.m20 + matrix1.m03 * matrix2.m30,
                matrix1.m00 * matrix2.m01 + matrix1.m01 * matrix2.m11 + matrix1.m02 * matrix2.m21 + matrix1.m03 * matrix2.m31,
                matrix1.m00 * matrix2.m02 + matrix1.m01 * matrix2.m12 + matrix1.m02 * matrix2.m22 + matrix1.m03 * matrix2.m32,
                matrix1.m00 * matrix2.m03 + matrix1.m01 * matrix2.m13 + matrix1.m02 * matrix2.m23 + matrix1.m03 * matrix2.m33,
                matrix1.m10 * matrix2.m00 + matrix1.m11 * matrix2.m10 + matrix1.m12 * matrix2.m20 + matrix1.m13 * matrix2.m30,
                matrix1.m10 * matrix2.m01 + matrix1.m11 * matrix2.m11 + matrix1.m12 * matrix2.m21 + matrix1.m13 * matrix2.m31,
                matrix1.m10 * matrix2.m02 + matrix1.m11 * matrix2.m12 + matrix1.m12 * matrix2.m22 + matrix1.m13 * matrix2.m32,
                matrix1.m10 * matrix2.m03 + matrix1.m11 * matrix2.m13 + matrix1.m12 * matrix2.m23 + matrix1.m13 * matrix2.m33,
                matrix1.m20 * matrix2.m00 + matrix1.m21 * matrix2.m10 + matrix1.m22 * matrix2.m20 + matrix1.m23 * matrix2.m30,
                matrix1.m20 * matrix2.m01 + matrix1.m21 * matrix2.m11 + matrix1.m22 * matrix2.m21 + matrix1.m23 * matrix2.m31,
                matrix1.m20 * matrix2.m02 + matrix1.m21 * matrix2.m12 + matrix1.m22 * matrix2.m22 + matrix1.m23 * matrix2.m32,
                matrix1.m20 * matrix2.m03 + matrix1.m21 * matrix2.m13 + matrix1.m22 * matrix2.m23 + matrix1.m23 * matrix2.m33,
                matrix1.m30 * matrix2.m00 + matrix1.m31 * matrix2.m10 + matrix1.m32 * matrix2.m20 + matrix1.m33 * matrix2.m30,
                matrix1.m30 * matrix2.m01 + matrix1.m31 * matrix2.m11 + matrix1.m32 * matrix2.m21 + matrix1.m33 * matrix2.m31,
                matrix1.m30 * matrix2.m02 + matrix1.m31 * matrix2.m12 + matrix1.m32 * matrix2.m22 + matrix1.m33 * matrix2.m32,
                matrix1.m30 * matrix2.m03 + matrix1.m31 * matrix2.m13 + matrix1.m32 * matrix2.m23 + matrix1.m33 * matrix2.m33);
    }
    friend Vector4 operator*(const Matrix4x4 &matrix, const Vector4 &vector) {
        return Vector4(
                matrix.m00 * vector.x + matrix.m01 * vector.y + matrix.m02 * vector.z + matrix.m03 * vector.w,
                matrix.m10 * vector.x + matrix.m11 * vector.y + matrix.m12 * vector.z + matrix.m13 * vector.w,
                matrix.m20 * vector.x + matrix.m21 * vector.y + matrix.m22 * vector.z + matrix.m23 * vector.w,
                matrix.m30 * vector.x + matrix.m31 * vector.y + matrix.m32 * vector.z + matrix.m33 * vector.w);
    }
    friend void operator*=(Vector4 &vector, const Matrix4x4 &matrix) {
        float x = matrix.m00 * vector.x + matrix.m01 * vector.y + matrix.m02 * vector.z + matrix.m03 * vector.w;
        float y = matrix.m10 * vector.x + matrix.m11 * vector.y + matrix.m12 * vector.z + matrix.m13 * vector.w;
        float z = matrix.m20 * vector.x + matrix.m21 * vector.y + matrix.m22 * vector.z + matrix.m23 * vector.w;
        float w = matrix.m30 * vector.x + matrix.m31 * vector.y + matrix.m32 * vector.z + matrix.m33 * vector.w;
        vector.x = x; vector.y = y; vector.z = z; vector.w = w;
    }
    friend Vector4 operator*(const Matrix4x4 *matrix, const Vector4 &vector) {
        return Vector4(
                matrix->m00 * vector.x + matrix->m01 * vector.y + matrix->m02 * vector.z + matrix->m03 * vector.w,
                matrix->m10 * vector.x + matrix->m11 * vector.y + matrix->m12 * vector.z + matrix->m13 * vector.w,
                matrix->m20 * vector.x + matrix->m21 * vector.y + matrix->m22 * vector.z + matrix->m23 * vector.w,
                matrix->m30 * vector.x + matrix->m31 * vector.y + matrix->m32 * vector.z + matrix->m33 * vector.w);
    }
    void print() const {
        std::printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",
                    m00, m01, m02, m03,
                    m10, m11, m12, m13,
                    m20, m21, m22, m23,
                    m30, m31, m32, m33);
    }
    Matrix4x4 transpose() const {
        return Matrix4x4(
                m00, m10, m20, m30,
                m01, m11, m21, m31,
                m02, m12, m22, m32,
                m03, m13, m23, m33);
    }
    Matrix4x4 inverse() const {
        float m2323 = m22 * m33 - m23 * m32 ;
        float m1323 = m21 * m33 - m23 * m31 ;
        float m1223 = m21 * m32 - m22 * m31 ;
        float m0323 = m20 * m33 - m23 * m30 ;
        float m0223 = m20 * m32 - m22 * m30 ;
        float m0123 = m20 * m31 - m21 * m30 ;
        float m2313 = m12 * m33 - m13 * m32 ;
        float m1313 = m11 * m33 - m13 * m31 ;
        float m1213 = m11 * m32 - m12 * m31 ;
        float m2312 = m12 * m23 - m13 * m22 ;
        float m1312 = m11 * m23 - m13 * m21 ;
        float m1212 = m11 * m22 - m12 * m21 ;
        float m0313 = m10 * m33 - m13 * m30 ;
        float m0213 = m10 * m32 - m12 * m30 ;
        float m0312 = m10 * m23 - m13 * m20 ;
        float m0212 = m10 * m22 - m12 * m20 ;
        float m0113 = m10 * m31 - m11 * m30 ;
        float m0112 = m10 * m21 - m11 * m20 ;

        float det = 1 / (
                + m00 * ( m11 * m2323 - m12 * m1323 + m13 * m1223 )
                - m01 * ( m10 * m2323 - m12 * m0323 + m13 * m0223 )
                + m02 * ( m10 * m1323 - m11 * m0323 + m13 * m0123 )
                - m03 * ( m10 * m1223 - m11 * m0223 + m12 * m0123 )) ;

        return Matrix4x4(
            det * +( m11 * m2323 - m12 * m1323 + m13 * m1223 ),
            det * -( m01 * m2323 - m02 * m1323 + m03 * m1223 ),
            det * +( m01 * m2313 - m02 * m1313 + m03 * m1213 ),
            det * -( m01 * m2312 - m02 * m1312 + m03 * m1212 ),
            det * -( m10 * m2323 - m12 * m0323 + m13 * m0223 ),
            det * +( m00 * m2323 - m02 * m0323 + m03 * m0223 ),
            det * -( m00 * m2313 - m02 * m0313 + m03 * m0213 ),
            det * +( m00 * m2312 - m02 * m0312 + m03 * m0212 ),
            det * +( m10 * m1323 - m11 * m0323 + m13 * m0123 ),
            det * -( m00 * m1323 - m01 * m0323 + m03 * m0123 ),
            det * +( m00 * m1313 - m01 * m0313 + m03 * m0113 ),
            det * -( m00 * m1312 - m01 * m0312 + m03 * m0112 ),
            det * -( m10 * m1223 - m11 * m0223 + m12 * m0123 ),
            det * +( m00 * m1223 - m01 * m0223 + m02 * m0123 ),
            det * -( m00 * m1213 - m01 * m0213 + m02 * m0113 ),
            det * +( m00 * m1212 - m01 * m0212 + m02 * m0112 )
        );
    }
    void multiply(float* X, float* Y, float* Z, float* W) {
        float x = m00 * *X + m01 * *Y + m02 * *Z + m03 * *W;
        float y = m10 * *X + m11 * *Y + m12 * *Z + m13 * *W;
        float z = m20 * *X + m21 * *Y + m22 * *Z + m23 * *W;
        float w = m30 * *X + m31 * *Y + m32 * *Z + m33 * *W;
        *X = x; *Y = y; *Z = z; *W = w;
    }
    inline void multiply(float* X, float* Y, float* Z) {
        float x = m00 * *X + m01 * *Y + m02 * *Z + m03;
        float y = m10 * *X + m11 * *Y + m12 * *Z + m13;
        float z = m20 * *X + m21 * *Y + m22 * *Z + m23;
        float w = m30 * *X + m31 * *Y + m32 * *Z + m33;
        *X = x; *Y = y; *Z = z;
    }
    static Matrix4x4 model(Vector4& t, Vector4& r, Vector4& s) {
        const float c3 = std::cos(r.z);
        const float s3 = std::sin(r.z);
        const float c2 = std::cos(r.x);
        const float s2 = std::sin(r.x);
        const float c1 = std::cos(r.y);
        const float s1 = std::sin(r.y);
        return Matrix4x4(
            s.x * (c1 * c3 + s1 * s2 * s3),
            s.x * (c2 * s3),
            s.x * (c1 * s2 * s3 - c3 * s1),
            0.0f,
            
            s.y * (c3 * s1 * s2 - c1 * s3),
            s.y * (c2 * c3),
            s.y * (c1 * c3 * s2 + s1 * s3),
            0.0f,
        
            s.z * (c2 * s1),
            s.z * (-s2),
            s.z * (c1 * c2),
            0.0f,
            
            t.x, t.y, t.z, 1.0f
        );
    }
};

#endif //MATRIX4X4
