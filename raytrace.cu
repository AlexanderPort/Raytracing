#include "raytrace.cuh"


#define ITERATIONS 50
#define THRESHOLD 100

/*

#define checkCudaErrors(value) checkCuda((value), #value, __FILE__, __LINE__)

void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) 
        << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset(); exit(99);
    }
}

*/

__global__ void __mandelbrot__(unsigned int* cbuffer, int w, int h, double dx, double dy,
                               double min_x, double max_x, double min_y, double max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = w * j + i;
    double real = 0, imag = 0;
    double cr = (2 * i / (double)w - 1) * (max_x - min_x) + min_x + dx;
    double ci = (2 * j / (double)h - 1) * (max_y - min_y) + min_y + dy;
    if ((i >= w) || (j >= h)) return;
    int iterations = 0;
    unsigned int color = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        double new_real = real * real - imag * imag + cr;
        double new_imag = 2 * real * imag + ci;
        real = new_real; imag = new_imag;
        if (real * real + imag * imag > THRESHOLD) {
            break;
        }
        iterations += 1;
    }
    color = ((double)iterations / ITERATIONS) * 0xffffff;
    cbuffer[index] = color;
}


__device__ linalg::vector3 ray_color(const linalg::vector3& ro, const linalg::vector3& rd, hitable** world, 
                                     curandState *random, unsigned int* environment_buffer, int ew, int eh) {
    hit_record record;
    linalg::vector3 cro = ro;
    linalg::vector3 crd = rd;
    linalg::vector3 cemitted = {0.0f, 0.0f, 0.0f};
    linalg::vector3 cattenuation{1.0f, 1.0f, 1.0f};
    for (int i = 0; i < 10; ++i) {
        if ((*world)->hit(cro, crd, 0.0f, FLT_MAX, record)) {
            linalg::vector3 attenuation;
            linalg::vector3 emitted = record.material->emitted(record.position);
            bool status = record.material->scatter(cro, crd, record, attenuation, cro, crd, random);
            if (status) { cattenuation *= attenuation; cemitted += (emitted * cattenuation); } 
            else { return cemitted + emitted * cattenuation; }
            
        } else {
            linalg::vector3 direction = linalg::normalize(crd);
        
            float u = 0.5f + atan2(direction.x, direction.z) / (2 * M_PI);
            float v = 0.5f - asin(direction.y) / M_PI;
            int index = int(u * (ew - 1)) + int(v * (eh - 1)) * ew;
            unsigned int color = environment_buffer[index];

            float r = ((color >> 16) & 0xFF) / 255.0f;
            float g = ((color >> 8) & 0xFF) / 255.0f;
            float b = ((color) & 0xFF) / 255.0f;
            
            linalg::vector3 diffuse_intensity{r, g, b};
            linalg::vector3 specular_intensity{r, g, b};
            hitable_list* w = (hitable_list*)(*world);
            for (int i = 0; i < w->num_lights; i++) {
                light* l = w->lights[i];
                linalg::vector3 direction = linalg::normalize(l->position - record.position);
                
                float distance = linalg::length(l->position - record.position);
                linalg::vector3 so = linalg::dot(direction, record.normal) < 0 ? 
                                     record.position - record.normal * 1e-3 : 
                                     record.position + record.normal * 1e-3;
                hit_record shadow;
                if ((*world)->hit(so, direction, 0.0f, FLT_MAX, shadow)) {
                    if (linalg::length(shadow.position - so) < distance) {
                        continue;
                    }
                }
                
                float reflection_power = linalg::dot(reflect(direction, record.normal), rd);
                specular_intensity += l->intensity * pow(max(0.0f, reflection_power), 2.0f);
                diffuse_intensity += l->intensity * max(0.0f, linalg::dot(direction, record.normal));
            }
            
            return cemitted + diffuse_intensity * specular_intensity * cattenuation;
        } 
    }
    return cemitted;
}


__global__ void render_init(int w, int h, int state, curandState *random) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= w) || (j >= h)) return;
    int index = w * j + i;
   
    curand_init(state * index, 0, 0, &random[index]);
}


__global__ void render(unsigned int* screen_buffer, int sw, int sh,
                       unsigned int* environment_buffer, int ew, int eh, 
                       float delta, float factor_x, float factor_y,
                       float camera_x, float camera_y, float camera_z,
                       float up_x, float up_y, float up_z,
                       float right_x, float right_y, float right_z,
                       float forward_x, float forward_y, float forward_z,
                       hitable** world, curandState *random) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= sw) || (j >= sh)) return;

    linalg::vector3 up = {up_x, up_y, up_z};
    linalg::vector3 right = {right_x, right_y, right_z};
    linalg::vector3 forward = {forward_x, forward_y, forward_z};

    //float dx = 2 * factor_x / sw / 4, dy = -2 * factor_y / sh / 4;
    //float ndc_x = (2 * ((i + 0.5f) / (float)sw) - 1) * factor_x;
    //float ndc_y = (1 - 2 * ((j + 0.5f) / (float)sh)) * factor_y;
    
    //linalg::vector3 ddx = dx * right + dy * up;
    //linalg::vector3 ddy = dy * right + dy * up;
    linalg::vector3 ro = {camera_x, camera_y, camera_z};
    //linalg::vector3 rd = ndc_x * right + ndc_y * up + forward;

    int pixel_index = sw * j + i;

    curandState local_random = random[pixel_index];
    /*
    linalg::vector3 color = ray_color(ro, linalg::normalize(rd), world, &local_random, environment_buffer, ew, eh);
    color += ray_color(ro, linalg::normalize(rd - ddy - ddx), world, &local_random, environment_buffer, ew, eh);
    color += ray_color(ro, linalg::normalize(rd + ddx - ddy), world, &local_random, environment_buffer, ew, eh);
    color += ray_color(ro, linalg::normalize(rd - ddx + ddy), world, &local_random, environment_buffer, ew, eh);
    color += ray_color(ro, linalg::normalize(rd + ddx + ddy), world, &local_random, environment_buffer, ew, eh);
    */
    
    linalg::vector3 color;
    float dd = 2 * curand_uniform(&local_random);
    float ndc_x = (2 * ((i + dd) / (float)sw) - 1) * factor_x;
    float ndc_y = (1 - 2 * ((j + dd) / (float)sh)) * factor_y;
    linalg::vector3 rd = linalg::normalize(ndc_x * right + ndc_y * up + forward);
    color += ray_color(ro, rd, world, &local_random, environment_buffer, ew, eh);

    random[pixel_index] = local_random;

    //if (color.x > 1.0f) color.x = 1.0f;
    //if (color.y > 1.0f) color.y = 1.0f;
    //if (color.z > 1.0f) color.z = 1.0f;

    color.x = sqrt(color.x);
    color.y = sqrt(color.y);
    color.z = sqrt(color.z);

    unsigned int c = screen_buffer[pixel_index];
    float R = ((c >> 16) & 0xFF) / 255.0f;
    float G = ((c >> 8) & 0xFF) / 255.0f;
    float B = ((c) & 0xFF) / 255.0f;

    unsigned int r = (unsigned int)(255 * (delta * color.x + (1 - delta) * R));
    unsigned int g = (unsigned int)(255 * (delta * color.y + (1 - delta) * G));
    unsigned int b = (unsigned int)(255 * (delta * color.z + (1 - delta) * B));
    screen_buffer[pixel_index] = (r << 16) | (g << 8) | b;
}

__global__ void init_world(hitable** objects, hitable** world, light** lights, int nx, int ny, curandState* random) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_random = random[0];
        int index = 0;
        for (size_t i = 1; i < 6; i++) {
            for (size_t j = 1; j < 6; j++) {
                material* m;
                if ((i * j) % 3 == 0) {
                    m = new metal({0.9f, 0.3f, 0.3f}, 0.99f);
                } else if ((i * j) % 3 == 1) {
                    m = new metal({0.3f, 0.9f, 0.3f}, 0.99f);
                } else if ((i * j) % 3 == 2) {
                    m = new metal({0.3f, 0.3f, 0.9f}, 0.99f);
                }
                objects[index++] = new sphere({i * 9, 3.0f, j * 9}, 2.5f, m);
            }
        }

        objects[index++] = new sphere({55, 0, 0}, 10, new metal({1.0f, 1.0f, 1.0f}));
        objects[index++] = new sphere({0, 0, 55}, 10, new metal({1.0f, 1.0f, 0.5f}));
        objects[index++] = new sphere({0, 0, 0}, 10, new metal({1.0f, 1.0f, 1.0f}));
        objects[index++] = new sphere({55, 0, 55}, 10, new metal({1.0f, 1.0f, 1.0f}));
        objects[index++] = new sphere({27, -100, 27}, 100, new metal({1.0f, 0.5f, 1.0f}, 0.8f));

        lights[0] = new light({100, 1000, 0}, {2.0f, 1.0f, 1.0f});

        *world = new hitable_list(objects, 30, lights, 0);

        random[0] = local_random;
    }
    
}

__global__ void free_world(hitable** list, hitable** world) {
    
}


namespace raytracing {
	void mandelbrot(unsigned int* screen_buffer, int nx, int ny, double dx, double dy, 
                  double min_x, double max_x, double min_y, double max_y) {
        //unsigned int* buffer;
        int tx = 8; int ty = 8;
        size_t num_pixels = nx * ny;
        size_t size = num_pixels * sizeof(unsigned int);

        if (screen_buffer == nullptr) cudaMalloc((void**)&screen_buffer, size);       
        
        dim3 blocks(nx / tx + 1, ny / ty + 1); dim3 threads(tx, ty);
        __mandelbrot__<<<blocks, threads>>>(screen_buffer, nx, ny, dx, dy, 
                                          min_x, max_x, min_y, max_y);

        cudaMemcpy(screen_buffer, screen_buffer, size, cudaMemcpyDeviceToHost);
	}
    
    void raytrace(unsigned int* screen_buffer, int sw, int sh,
                  unsigned int* environment_buffer, int ew, int eh,
                  float delta, const Camera& camera) {
        int tx = 8; int ty = 8;
        size_t random_size = sw * sh * sizeof(curandState);
        size_t screen_buffer_size = sw * sh * sizeof(unsigned int);
        size_t environment_buffer_size = ew * eh * sizeof(unsigned int);

        dim3 blocks(sw / tx + 1, sh / ty + 1); dim3 threads(tx, ty);

        if (!init) {
            cudaMalloc((void**)&random, random_size);
            cudaMalloc((void**)&cu_world, sizeof(hitable*));
            cudaMalloc((void**)&cu_screen_buffer, screen_buffer_size);
            cudaMalloc((void**)&cu_lights, cu_num_lights * sizeof(light*));
            cudaMalloc((void**)&cu_objects, cu_num_objects * sizeof(hitable*));
            cudaMalloc((void**)&cu_environment_buffer, environment_buffer_size);
            
            cudaMemcpy(cu_environment_buffer, environment_buffer, environment_buffer_size, cudaMemcpyHostToDevice);

            render_init<<<blocks, threads>>>(sw, sh, state, random); state++;

            init_world<<<1,1>>>(cu_objects, cu_world, cu_lights, sw, sh, random);

            init = true;
        };       

        cudaMemcpy(cu_screen_buffer, screen_buffer, screen_buffer_size, cudaMemcpyHostToDevice);

        render_init<<<blocks, threads>>>(sw, sh, state, random); state++;

        render<<<blocks, threads>>>(
            cu_screen_buffer, sw, sh, 
            cu_environment_buffer, ew, eh,
            delta, camera.factor_x, camera.factor_y,
            camera.position.x, camera.position.y,  camera.position.z, 
            camera.up.x, camera.up.y, camera.up.z,
            camera.right.x, camera.right.y, camera.right.z,
            camera.forward.x, camera.forward.y, camera.forward.z,
            cu_world, random);

        cudaMemcpy(screen_buffer, cu_screen_buffer, screen_buffer_size, cudaMemcpyDeviceToHost);
	}
    void free() {
        cudaFree(random);
        cudaFree(cu_world);
        cudaFree(cu_lights);
        cudaFree(cu_objects);
        cudaFree(cu_screen_buffer);
        cudaFree(cu_environment_buffer);
        cudaDeviceReset();
    }
}








/*
__device__ unsigned int RGBtoHEX(unsigned int r, unsigned int g, unsigned int b) {
    return (r << 16) | (g << 8) | b;
}
__device__ float get_distance(const linalg::vector3& position, const linalg::vector3& point) {
    float x = point.x - position.x;
    float y = point.y - position.y;
    float z = point.z - position.z;
    float radius = 1;
    x = sin(x); y = sin(y); z = sin(z);
    float sphere = (sqrt(x * x + y * y + z * z)) - radius / 2 - 0.1;

    float cube = max(max(abs(x), abs(y)), abs(z)) - radius / 2;

    return max(-sphere, cube);
}

__device__ float mandelbulb(const linalg::vector3& point) {
    linalg::vector3 z = point;
    float power = 10;
    float dr = 1; 
    float r = 0;
    int i;
    for (i = 0; i < 300; i++) {
        r = linalg::length(z);
        if (r > 2) { break; }
        float theta = acos(z.z / r);
        float phi = atan2(z.y, z.x);
        dr = pow(r, power - 1.0f) * power * dr + 1.0;
        float zr = pow(r, power);
        theta = theta * power;
        phi = phi * power;
        z = linalg::vector3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
        z = zr * z + point;
    }
    return 0.5 * log(r) * r / dr;
}

__device__ unsigned int raymarch(const linalg::vector3& origin, const linalg::vector3& direction) {
    float DISTANCE = 0;
    float MINIMUM_HIT_DISTANCE = 0.0005;
    float MAXIMUM_TRACE_DISTANCE = 300;
    for (int i = 0; i < ITERATIONS; ++i) {
        linalg::vector3 current = origin + DISTANCE * direction;
        float distance = mandelbulb(current);
        float d = float(i + 1) / ITERATIONS;
        if (distance < MINIMUM_HIT_DISTANCE) {
            int r = 255 * (1 - d * 2);
            int g = 0;
            int b = 255 * d;
            if (r > 255) r = 255;
            if (b > 255) b = 255;
            return RGBtoHEX(b, b, b);

            //return 256 * (1 - float(i) / ITERATIONS);
        }
        
        if (DISTANCE > MAXIMUM_TRACE_DISTANCE) {
            int r = 255 * (1 - d * 2);
            int g = 0;
            int b = 255 * d;
            if (r > 255) r = 255;
            if (b > 255) b = 255;
            return RGBtoHEX(r, g, b);
        }
        DISTANCE += distance;
    }
    return 0x000000;
}
*/