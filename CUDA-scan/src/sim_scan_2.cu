#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data_structures.h"
#include "utils.cuh"

// 记录各个表扫描到的总数据量
__device__ int scanCustomerCounter = 0;
__device__ int scanOrderCounter = 0;
__device__ int scanLineitemCounter = 0;

/**
 * @brief GPU 加速扫描 Customer 表
 *
 * @param data 指向 `Customer` 对象数组的指针，表示要扫描的数据。
 * @param size 表示 `Customer` 对象数组的大小，即数据的元素总数。
 *
 * @details
 * 每个线程处理一个 `Customer` 对象，通过其索引计算每个线程应该处理的数据。
 * 计算每个 `Customer` 对象的 `C_CUSTKEY` 值的总和，并将每个线程的计数结果通过原子操作累加到全局计数器 `scanCustomerCounter` 中。
 */
__global__ void scanCustomer(Customer *data, int size)
{
    /*
    作用: 计算每一个线程在grid 中的唯一全局索引
    解释:
    1. threadIdx.x: 这是当前线程在其线程块中的索引。假设每个线程块包含 512 个线程，则 threadIdx.x 的值范围是从 0 到 511。
    2. blockIdx.x: 这是当前线程块在网格中的索引。假设网格中有多个线程块，则 blockIdx.x 的值范围取决于网格中线程块的数量。
    3. blockDim.x: 这是每个线程块中线程的数量。设每个线程块包含 512 个线程，即 blockDim.x 的值为 512。

    例子:
    假设你要对一个长度为 1024 的数组进行操作，并且你使用了 2 个线程块(Block)，每个线程块包含 512 个线程。如何为每一个线程计算其全局索引的:
    - 设置线程块数量 (numBlocks): 2
    - 每个线程块中的线程数量 (blockSize): 512

    计算线程块 0 中的线程：对于 blockIdx.x = 0:
        threadIdx.x = 0 时，idx = 512 * 0 + 0 = 0
        threadIdx.x = 1 时，idx = 512 * 0 + 1 = 1
        ...
        threadIdx.x = 511 时，idx = 512 * 0 + 511 = 511

    线程块1中的线程 对于 blockIdx.x = 1:
        threadIdx.x = 0 时，idx = 512 * 1 + 0 = 512
        threadIdx.x = 1 时，idx = 512 * 1 + 1 = 513
        ...
        threadIdx.x = 511 时，idx = 512 * 1 + 511 = 1023
*/
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size)
        return;

    int localCounter = 0;
    int localKeySum = 0;

    // Some computing task
    localKeySum += data[idx].C_CUSTKEY;
    localCounter += 1;

    atomicAdd(&scanCustomerCounter, localCounter);
}

/**
 * @brief GPU 加速扫描 Orders 表
 *
 * @param data 指向 `Orders` 对象数组的指针，表示要扫描的数据。
 * @param size 表示 `Orders` 对象数组的大小，即数据的元素总数。
 *
 * @details
 * 每个线程处理一个 `Orders` 对象，通过其索引计算每个线程应该处理的数据。
 * 计算每个 `Orders` 对象的 `O_TOTALPRICE` 值的总和;
 * 并将每个线程的计数结果通过原子操作累加到全局计数器 `scanOrderCounter` 中。
 */
__global__ void scanOrders(Orders *data, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= size)
        return;

    int localCounter = 0;
    int localPriceSum = 0;

    // Some computing task
    localPriceSum += (int)data[idx].O_TOTALPRICE;
    localCounter += 1;

    atomicAdd(&scanOrderCounter, localCounter);
}

/**
 * @brief GPU 加速扫描 Lineitems 表
 *
 * @param data 指向 `Lineitem` 对象数组的指针，表示要扫描的数据。
 * @param size 表示 `Lineitem` 对象数组的大小，即数据的元素总数。
 *
 * @details
 * 每个线程处理一个 `Lineitem` 对象，通过其索引计算每个线程应该处理的数据。
 * 计算每个 `Lineitem` 对象的 `L_QUANTITY` 值的总和，
 * 并将每个线程的计数结果通过原子操作累加到全局计数器 `scanLineitemCounter` 中。
 */
__global__ void scanLineitems(Lineitem *data, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size)
        return;

    // Some computing task
    int localCounter = 0;
    int localQualitySum = 0;

    localQualitySum += data[idx].L_QUANTITY;
    localCounter += 1;

    atomicAdd(&scanLineitemCounter, localCounter);
}

// 输出各个表扫描到的总数据量
template <typename T>
void checkScanResults(const T &deviceCounter, const char *tableName)
{
    int hostCounter;
    // 从设备内存中复制计数器值到主机内存
    checkCudaError(cudaMemcpyFromSymbol(&hostCounter, deviceCounter, sizeof(int), 0, cudaMemcpyDeviceToHost));
    // 输出扫描得到的结果，包括表名
    std::cout << "Total scanned elements [Table-" << tableName << "]: " << hostCounter << std::endl;
}

int main()
{
    injectData();

    // 计算 BlockSize; BlockNums
    dim3 blockSize(512);

    // 定义每次扫描表的 block 数目
    dim3 numBlocksCustomers((customers.size() + blockSize.x - 1) / blockSize.x);
    dim3 numBlocksOrders((orders.size() + blockSize.x - 1) / blockSize.x);
    dim3 numBlocksLineitems((lineitems.size() + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalMilliseconds = 0;
    float milliseconds = 0;

    Customer *dev_customers = nullptr;
    Orders *dev_orders = nullptr;
    Lineitem *dev_lineitems = nullptr;

    // Record start time
    cudaEventRecord(start);

    allocateAndLaunchKernel(
        (void *&)dev_customers,
        customers.data(),
        customers.size(),
        sizeof(Customer),
        blockSize,
        numBlocksCustomers,
        (void (*)(void *, int))scanCustomer);

    allocateAndLaunchKernel(
        (void *&)dev_orders,
        orders.data(),
        orders.size(),
        sizeof(Orders),
        blockSize,
        numBlocksOrders,
        (void (*)(void *, int))scanOrders);

    allocateAndLaunchKernel(
        (void *&)dev_lineitems,
        lineitems.data(),
        lineitems.size(),
        sizeof(Lineitem),
        blockSize,
        numBlocksLineitems,
        (void (*)(void *, int))scanLineitems);

    // Record stop time and calculate total time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totalMilliseconds += milliseconds;

    checkScanResults(scanCustomerCounter, "Customer");
    checkScanResults(scanOrderCounter, "Order");
    checkScanResults(scanLineitemCounter, "LineItem");

    std::cout << "[Scan]Total time taken by GPU for all scans: " << totalMilliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}