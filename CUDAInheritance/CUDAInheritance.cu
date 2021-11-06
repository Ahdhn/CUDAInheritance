#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "gtest/gtest.h"

#include "helper.h"

class Base
{
   public:
    __device__ __host__ Base() : m_data(0)
    {
    }

    __device__ __host__ __forceinline__ int get_data() const
    {
        return m_data;
    }

    __device__ __host__ __forceinline__ void set_data1(int d)
    {
        m_data = d;
    }

    virtual ~Base() = default;

    int m_data;
};

class Derived : public Base
{
   public:
    __device__ __host__ Derived(){};

    __device__ __host__ __forceinline__ int get_data() const
    {
        return this->m_data;
    };

    __device__ __host__ __forceinline__ void set_data(int d, bool base)
    {
        if (base) {
            this->set_data1(d);
        } else {
            m_d = d;
        }
    };

    virtual ~Derived() = default;

    int m_d;
};

__global__ void kernel(Derived d, const int data, const bool base)
{
    d.set_data(data, base);
}

TEST(CUDAInheritance, Test0)
{
    Derived      d;
    int          val = rand();
    int          num_run = 1E6;
    cudaStream_t stream;
    CUDA_ERROR(cudaStreamCreate(&stream));

    std::cout << "Accessing base's method through derived class --- ";
    CUDATimer timer;
    timer.start(stream);
    for (int n = 0; n < num_run; ++n) {
        kernel<<<1, 1, 0, stream>>>(d, val, true);
    }
    timer.stop();
    std::cout << " time = " << timer.elapsed_millis() << " (ms)\n";

    CUDA_ERROR(cudaDeviceSynchronize());

    std::cout << "Accessing only derived class methods --- ";
    val = rand();
    timer.start(stream);
    for (int n = 0; n < num_run; ++n) {
        kernel<<<1, 1, 0, stream>>>(d, val, false);
    }
    timer.stop();
    std::cout << " time = " << timer.elapsed_millis() << " (ms)\n";

    cudaError_t status = cudaDeviceSynchronize();
    EXPECT_EQ(status, cudaSuccess);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
