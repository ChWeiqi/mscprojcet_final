#include <iostream>
#include <vector>
#include <chrono>
#include "data_structures.h"
#include "cuda_runtime.h"

// 数据行计数器
int scanCustomerCounter = 0;
int scanOrderCounter = 0;
int scanLineitemCounter = 0;

void scanCustomers(const std::vector<Customer> &customers)
{
    int totalCustKey = 0;
    for (size_t idx = 0; idx < customers.size(); ++idx)
    {
        totalCustKey += customers[idx].C_CUSTKEY;
        scanCustomerCounter++;
    }
}

void scanOrders(const std::vector<Orders> &orders)
{
    double totalOrderPrice = 0;

    for (size_t idx = 0; idx < orders.size(); ++idx)
    {
        totalOrderPrice += orders[idx].O_TOTALPRICE;
        scanOrderCounter++;
    }
}

void scanLineitems(const std::vector<Lineitem> &lineitems)
{
    int totalQuantity = 0;

    for (size_t idx = 0; idx < lineitems.size(); ++idx)
    {
        totalQuantity += lineitems[idx].L_QUANTITY;
        scanLineitemCounter++;
    }
}

template <typename T>
void checkScanResults(const T hostCounter, const char *tableName)
{
    std::cout << "Total scanned elements [Table-" << tableName << "]: " << hostCounter << std::endl;
}

int main()
{
    injectData();

    auto start = std::chrono::high_resolution_clock::now();

    scanCustomers(customers);
    scanOrders(orders);
    scanLineitems(lineitems);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    // 输出每个表的扫描结果
    checkScanResults(scanCustomerCounter, "Customer");
    checkScanResults(scanOrderCounter, "Order");
    checkScanResults(scanLineitemCounter, "LineItem");

    std::cout << "[Scan] Total time taken by CPU for all scans: " << duration.count() << " ms" << std::endl;

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
