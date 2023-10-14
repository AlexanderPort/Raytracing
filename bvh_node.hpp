#ifndef BVH_HPP
#define BVH_HPP

#include <algorithm>

#include "hitable.hpp"
#include "hitable_list.hpp"


enum Axis { X, Y, Z };


static __host__ __device__ void swap(hitable*& p1, hitable*& p2) {
    hitable* temp = p1;
    *p1 = *p2;
    p2 = temp;
}


template<Axis axis>
static __host__ __device__ void bubble_sort(hitable** objects, int num_objects) {
    for (int i = 0; i < num_objects - 1; i++) {
        for (int j = 0; j < num_objects - i - 1; j++) {
            aabb box_left, box_right;
            hitable *ah = objects[j + 0];
            hitable *bh = objects[j + 1];
            
            ah->bounding_box(box_left);
            bh->bounding_box(box_right);

            if ((axis == X && (box_left.minimum.x - box_right.minimum.x) < 0.0f)
             || (axis == Y && (box_left.minimum.y - box_right.minimum.y) < 0.0f)
             || (axis == Z && (box_left.minimum.z - box_right.minimum.z) < 0.0f)) {
                swap(objects[j + 0], objects[j + 1]);
            }
        }
    }
}


class bvh_node {
public:
    __host__ __device__ bvh_node() {}
    __host__ bvh_node::bvh_node(int index) {
        this->index = index;
        this->left = 2 * index + 1;
        this->right = 2 * index + 2;
    }

    static __host__ void build(hitable** objects, int num_objects, bvh_node** nodes, int index) {
        int axis = rand() % 4;

        if (axis == 0) bubble_sort<X>(objects, num_objects);
        else if (axis == 1) bubble_sort<Y>(objects, num_objects);
        else bubble_sort<Z>(objects, num_objects);

        bvh_node* node = new bvh_node(index);

        if (num_objects == 1) {
            node->object = objects[0];
            node->left = -1; node->right = -1;
            node->object->bounding_box(node->box);
        } else {
            bvh_node::build(objects, num_objects / 2, nodes, 2 * index + 1);
            bvh_node::build(objects + num_objects / 2, num_objects - num_objects / 2, nodes, 2 * index + 2);
        }

        node->box = aabb::surrounding_box(nodes[node->left]->box, nodes[node->right]->box); nodes[index] = node;
    }

    static __device__ bool hit(const bvh_node** nodes,
                               linalg::vector3& ro, const linalg::vector3& rd, 
                               float tmin, float tmax, hit_record& record) {
        return true;
    }

    __host__ __device__ bool bounding_box(aabb& box) const {
        box = this->box; return true;
    }
    int left;
    int right;
    int index;
    aabb box;
    hitable* object = nullptr;
};



#endif