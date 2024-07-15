#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data_structures.h"
#include "utils.cuh"

/**
 * @brief 计算每个客户的订单总金额，并将结果存储在 customerOrderTotals 数组中。
 *
 * @param customers 指向 `Customer` 对象数组的指针，表示所有客户数据。
 * @param orders 指向 `Orders` 对象数组的指针，表示所有订单数据。
 * @param lineitems 指向 `Lineitem` 对象数组的指针，表示所有订单项数据。
 * @param sizeCust 客户数组的大小，即客户的总数。
 * @param sizeOrders 订单数组的大小，即订单的总数。
 * @param sizeLineitems 订单项数组的大小，即订单项的总数。
 * @param customerOrderTotals 指向浮点数组的指针，用于存储每个客户的订单总金额。
 *
 * @details
 * 每个线程处理一个客户，计算该客户所有相关订单的总金额。
 * 1. 遍历所有订单，找到与当前客户相关的订单。
 * 2. 对于每个订单，累加其总金额，并累加所有相关订单项的数量乘以10的值。
 * 3. 将累加结果存储在 customerOrderTotals 数组对应位置。
 */
__global__ void scanCustomerOrders(
    Customer *customers,
    Orders *orders,
    Lineitem *lineitems,
    int sizeCust,
    int sizeOrders,
    int sizeLineitems,
    float *customerOrderTotals)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= sizeCust)
        return;

    float total = 0;
    for (int i = 0; i < sizeOrders; i++)
    {
        if (orders[i].O_CUSTKEY == customers[idx].C_CUSTKEY)
        {
            double orderTotal = orders[i].O_TOTALPRICE;
            for (int j = 0; j < sizeLineitems; j++)
            {
                if (lineitems[j].L_ORDERKEY == orders[i].O_ORDERKEY)
                {
                    orderTotal += lineitems[j].L_QUANTITY * 10;
                }
            }
            total += orderTotal;
        }
    }
    customerOrderTotals[idx] = total;
}

/**
 * @brief 分配设备内存、复制数据并启动 CUDA 核函数 `scanCustomerOrders`。
 *
 * @param hostCustomers 指向主机端 `Customer` 对象数组的指针。
 * @param numCustomers 客户数组的大小，即客户的总数量。
 * @param hostOrders 指向主机端 `Orders` 对象数组的指针。
 * @param numOrders 订单数组的大小，即订单的总数量。
 * @param hostLineitems 指向主机端 `Lineitem` 对象数组的指针。
 * @param numLineitems 订单项数组的大小，即订单项的总数量。
 * @param hostCustomerOrderTotals 指向浮点数组指针的指针，用于存储每个客户的订单总金额，该数组将在主机端分配。
 * @param blockSize 每个线程块中的线程数。
 * @param numBlocks 网格中线程块的数量。
 *
 * @details
 * 1. 使用 `cudaMalloc` 在设备端分配内存，并使用 `cudaMemcpy` 将主机数据复制到设备。
 * 2. 启动 CUDA 核函数 `scanCustomerOrders` 以计算每个客户的订单总金额。
 * 3. 同步设备并将结果从设备复制到主机。
 * 4. 释放设备内存。
 */
void allocateAndLaunchKernel(
    Customer *hostCustomers, int numCustomers,
    Orders *hostOrders, int numOrders,
    Lineitem *hostLineitems, int numLineitems,
    float **hostCustomerOrderTotals,
    dim3 blockSize,
    dim3 numBlocks)
{
    Customer *devCustomers = nullptr;
    Orders *devOrders = nullptr;
    Lineitem *devLineitems = nullptr;
    float *devCustomerOrderTotals = nullptr;

    // 分配设备内存并将客户数据从主机复制到设备
    checkCudaError(cudaMalloc(&devCustomers, numCustomers * sizeof(Customer)));
    checkCudaError(cudaMemcpy(devCustomers, hostCustomers, numCustomers * sizeof(Customer), cudaMemcpyHostToDevice));

    // 分配设备内存并将订单数据从主机复制到设备
    checkCudaError(cudaMalloc(&devOrders, numOrders * sizeof(Orders)));
    checkCudaError(cudaMemcpy(devOrders, hostOrders, numOrders * sizeof(Orders), cudaMemcpyHostToDevice));

    // 分配设备内存并将订单项数据从主机复制到设备
    checkCudaError(cudaMalloc(&devLineitems, numLineitems * sizeof(Lineitem)));
    checkCudaError(cudaMemcpy(devLineitems, hostLineitems, numLineitems * sizeof(Lineitem), cudaMemcpyHostToDevice));

    // 分配设备内存用于存储每个客户的订单总金额
    checkCudaError(cudaMalloc(&devCustomerOrderTotals, numCustomers * sizeof(float)));

    // 执行核函数
    scanCustomerOrders<<<numBlocks, blockSize>>>(
        devCustomers,
        devOrders,
        devLineitems,
        numCustomers,
        numOrders,
        numLineitems,
        devCustomerOrderTotals);

    checkCudaError(cudaDeviceSynchronize());

    // 分配主机内存用于存储从设备复制回来的订单总金额
    *hostCustomerOrderTotals = new float[numCustomers];
    checkCudaError(cudaMemcpy(*hostCustomerOrderTotals, devCustomerOrderTotals, numCustomers * sizeof(float), cudaMemcpyDeviceToHost));

    // 释放设备内存
    cudaFree(devCustomers);
    cudaFree(devOrders);
    cudaFree(devLineitems);
    cudaFree(devCustomerOrderTotals);
}

int main()
{
    // 模拟数据注入
    // injectData(15000, 150000, 1000);
    injectData(1500000, 150000, 1000);

    // int blockSizes = 512;    //Initial test CUDA and Cpp

    // 测试不同的 blockSize；找到最佳取值
    // 本地 laptop GPU device: NVIDIA GeForce RTX 4060 Laptop GPU
    // 查询设备信息之后的最大支持: Maximum number of threads per thread block:1024
    int blockSizes[] = {128, 256, 512, 1024, 2048};
    float bestTime = FLT_MAX;
    int bestBlockSize = 0;

    for (int size : blockSizes)
    {
        dim3 blockSize(size);
        dim3 numBlocks((customers.size() + blockSize.x - 1) / blockSize.x);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalMilliseconds = 0;
        float *customerOrderTotals = nullptr;

        cudaEventRecord(start);
        allocateAndLaunchKernel(
            customers.data(),
            customers.size(),
            orders.data(),
            orders.size(),
            lineitems.data(),
            lineitems.size(),
            &customerOrderTotals,
            blockSize,
            numBlocks);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // 计算用时
        cudaEventElapsedTime(&totalMilliseconds, start, stop);

        std::cout << "Block Size: " << size << ", Total time taken by GPU: " << totalMilliseconds << " ms\n";

        if (totalMilliseconds < bestTime)
        {
            bestTime = totalMilliseconds;
            bestBlockSize = size;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        delete[] customerOrderTotals;
    }

    std::cout << "Best block size: " << bestBlockSize << " with minimum time: " << bestTime << " ms\n";
    return 0;
}