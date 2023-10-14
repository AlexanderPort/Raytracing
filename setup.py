x = '''

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
'''

x = x.replace('matrix1', 'f1').replace('matrix2', 'f2').replace('m', 'f')
print(x)