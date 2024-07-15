#include <iostream>
#include <vector>
#include <chrono>
#include "data_structures.h"

// C++ 实现多表扫描聚合和简单计算
void scanCustomerOrdersCPU(
    const std::vector<Customer> &customers,
    const std::vector<Orders> &orders,
    const std::vector<Lineitem> &lineitems,
    std::vector<float> &customerOrderTotals)
{
    for (int idx = 0; idx < customers.size(); ++idx)
    {
        float total = 0.0;
        for (int i = 0; i < orders.size(); i++)
        {
            if (orders[i].O_CUSTKEY == customers[idx].C_CUSTKEY)
            {
                double orderTotal = orders[i].O_TOTALPRICE;
                for (int j = 0; j < lineitems.size(); j++)
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
}

// CPU 执行与 CUDA 相同的scan逻辑
void allocateAndLaunchCPUKernel(
    const std::vector<Customer> &hostCustomers,
    const std::vector<Orders> &hostOrders,
    const std::vector<Lineitem> &hostLineitems,
    std::vector<float> &hostCustomerOrderTotals)
{
    hostCustomerOrderTotals.resize(hostCustomers.size(), 0.0f);
    scanCustomerOrdersCPU(hostCustomers, hostOrders, hostLineitems, hostCustomerOrderTotals);
}

int main()
{
    // injectData(15000, 150000, 1000);
    injectData(1500000, 150000, 1000);

    std::vector<float> customerOrderTotals(customers.size(), 0.0f);

    auto startTime = std::chrono::high_resolution_clock::now();
    allocateAndLaunchCPUKernel(customers, orders, lineitems, customerOrderTotals);
    auto endTime = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double, std::milli> totalMilliseconds = endTime - startTime;

    std::cout << "[Scan] Total time taken by CPU for scan and aggregation: " << totalMilliseconds.count() << " ms\n";

    return 0;
}