
// 友好打印错误信息
void checkCudaError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/**
 * @brief 分配设备内存、复制数据并启动 CUDA 核函数。
 *
 * @param dev_data 引用，用于存储分配到的设备内存指针。
 * @param host_data 指向主机数据的指针，数据将从该位置复制到设备内存。
 * @param count 数据元素的数量。
 * @param elem_size 单个数据元素的大小（字节）。
 * @param blockSize 每个线程块中的线程数。
 * @param numBlocks 网格中线程块的数量。
 * @param kernel 指向 CUDA 核函数的指针，该函数将在分配和数据复制完成后被调用。
 *
 * @details
 * 1. 使用 `cudaMalloc` 分配设备内存。
 * 2. 使用 `cudaMemcpy` 将主机数据复制到设备内存。
 * 3. 使用 `kernel<<<numBlocks, blockSize>>>` 启动 CUDA 核函数。
 * 4. 使用 `cudaFree` 释放设备内存。
 */
void allocateAndLaunchKernel(
    void *&dev_data,
    const void *host_data,
    size_t count,
    size_t elem_size,
    dim3 blockSize,
    dim3 numBlocks,
    void (*kernel)(void *, int))
{
    checkCudaError(cudaMalloc(&dev_data, count * elem_size));
    checkCudaError(cudaMemcpy(dev_data, host_data, count * elem_size, cudaMemcpyHostToDevice));

    kernel<<<numBlocks, blockSize>>>((decltype(dev_data))dev_data, count);

    cudaFree(dev_data);
}