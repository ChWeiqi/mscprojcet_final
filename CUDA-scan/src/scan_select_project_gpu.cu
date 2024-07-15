#include "data_structures.h"
#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cuh"

/**
 *  模拟 SQL:
 *  SELECT C.C_NAME, O.O_ORDERDATE, L.L_QUANTITY
    FROM Customer C
    JOIN Orders O ON C.C_CUSTKEY = O.O_CUSTKEY
    JOIN Lineitem L ON O.O_ORDERKEY = L.L_ORDERKEY
    WHERE O.O_TOTALPRICE > threshold;
 */

/**
 * 处理查询并统计满足特定条件的订单数量。
 *
 * @param customers 指向 Customer 结构体数组的指针，包含所有客户的信息。
 * @param numCustomers 整数，表示 customers 数组中的元素数量。
 * @param orders 指向 Orders 结构体数组的指针，包含所有订单的信息。
 * @param numOrders 整数，表示 orders 数组中的元素数量。
 * @param lineitems 指向 Lineitem 结构体数组的指针，包含所有订单项的信息。
 * @param numLineitems 整数，表示 lineitems 数组中的元素数量。
 * @param threshold 双精度浮点数，用于设置订单总价的筛选阈值。
 * @param resultCount 指向整数的指针，用于存储满足条件的订单数量。
 *
 * 此函数将遍历所有订单，并检查它们的总价是否超过了 threshold。对于每个满足条件的订单，
 * 该函数将增加存储在 resultCount 指针所指向的值。
 */
__global__ void processQuery(
    Customer *customers,
    int numCustomers,
    Orders *orders,
    int numOrders,
    Lineitem *lineitems,
    int numLineitems,
    double threshold,
    int *resultCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCustomers)
        return;

    int count = 0;
    for (int i = 0; i < numOrders; i++)
    {
        if (customers[idx].C_CUSTKEY == orders[i].O_CUSTKEY && orders[i].O_TOTALPRICE > threshold)
        {
            for (int j = 0; j < numLineitems; j++)
            {
                if (orders[i].O_ORDERKEY == lineitems[j].L_ORDERKEY)
                {
                    count++;
                    // For simplicity, we just count the matches. In production, you might store results.
                    printf("------ [Customer Name: %s, Order Date: %s, Quantity: %d ] ------\n", customers[idx].C_NAME, orders[i].O_ORDERDATE, lineitems[j].L_QUANTITY);
                }
            }
        }
    }
    atomicAdd(resultCount, count);
}

void allocateAndLaunchQueryKernel(
    Customer *hostCustomers, int numCustomers,
    Orders *hostOrders, int numOrders,
    Lineitem *hostLineitems, int numLineitems,
    double threshold,
    int *hostResultCount,
    dim3 blockSize,
    dim3 numBlocks)
{
    Customer *devCustomers = nullptr;
    Orders *devOrders = nullptr;
    Lineitem *devLineitems = nullptr;
    int *devResultCount = nullptr;

    // 分配设备内存并复制数据
    cudaMalloc(&devCustomers, numCustomers * sizeof(Customer));
    cudaMemcpy(devCustomers, hostCustomers, numCustomers * sizeof(Customer), cudaMemcpyHostToDevice);

    cudaMalloc(&devOrders, numOrders * sizeof(Orders));
    cudaMemcpy(devOrders, hostOrders, numOrders * sizeof(Orders), cudaMemcpyHostToDevice);

    cudaMalloc(&devLineitems, numLineitems * sizeof(Lineitem));
    cudaMemcpy(devLineitems, hostLineitems, numLineitems * sizeof(Lineitem), cudaMemcpyHostToDevice);

    cudaMalloc(&devResultCount, sizeof(int));
    cudaMemset(devResultCount, 0, sizeof(int)); // 初始化结果计数为0

    // 执行核函数
    processQuery<<<numBlocks, blockSize>>>(
        devCustomers,
        numCustomers,
        devOrders,
        numOrders,
        devLineitems,
        numLineitems,
        threshold,
        devResultCount);

    cudaDeviceSynchronize(); // 确保所有线程完成

    // 复制结果回主机
    cudaMemcpy(hostResultCount, devResultCount, sizeof(int), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(devCustomers);
    cudaFree(devOrders);
    cudaFree(devLineitems);
    cudaFree(devResultCount);
}

int main()
{
    // 示例使用数据和调用函数（省略数据初始化）
    injectData(1500000, 150000, 1000);
    int hostResultCount = 0;
    double threshold = 1000.0;

    // 测试不同的 blockSize；找到最佳取值
    // 本地 laptop GPU device: NVIDIA GeForce RTX 4060 Laptop GPU
    // 查询设备信息之后的最大支持: Maximum number of threads per thread block:1024
    int blockSizes[] = {128, 256, 512, 1024 /*, 2048*/};
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
        allocateAndLaunchQueryKernel(
            customers.data(),
            customers.size(),
            orders.data(),
            orders.size(),
            lineitems.data(),
            lineitems.size(),
            threshold,
            &hostResultCount,
            blockSize,
            numBlocks);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // 计算用时
        cudaEventElapsedTime(&totalMilliseconds, start, stop);

        std::cout << "Block Size: " << size << ", Total time taken by GPU: " << totalMilliseconds << " ms\n";

        // std::cout << "Number of results: " << hostResultCount << std::endl;

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