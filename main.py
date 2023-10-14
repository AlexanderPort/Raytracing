import os



INCLUDES = [
        "-IC:/Libs/SDL2-2.0.14/include",
        "-IC:/Libs/SDL2_image-2.0.5/include",
    ]
LIBRARIES = [
    "-LC:/Libs/SDL2-2.0.14/lib/x64",
    "-LC:/Libs/SDL2_image-2.0.5/lib/x64",
]

INCLUDES, LIBRARIES = ' '.join(INCLUDES), ' '.join(LIBRARIES)
files = 'hitable.cpp hitable_list.cpp aabb.cpp bvh_node.cpp material.cpp sphere.cpp plane.cpp light.cpp linalg.cpp camera.cpp raytrace.cu main.cpp'
os.system(f'nvcc {INCLUDES} {LIBRARIES} -gencode arch=compute_75,code=sm_75 {files} -o raytrace.exe -lSDL2main -lSDL2 -lSDL2_Image')

# os.system('g++ -LC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/lib/x64 main.cpp -o main.exe raytrace.o -lcuda -lcudart')