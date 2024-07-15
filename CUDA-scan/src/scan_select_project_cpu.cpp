#include <iostream>
#include <vector>
#include <chrono>
#include "data_structures.h"

void processQueryCPU(
    std::vector<Customer> &customers,
    std::vector<Orders> &orders,
    std::vector<Lineitem> &lineitems,
    double threshold,
    int &resultCount)
{
    resultCount = 0;
    for (const auto &customer : customers)
    {
        for (const auto &order : orders)
        {
            if (customer.C_CUSTKEY == order.O_CUSTKEY && order.O_TOTALPRICE > threshold)
            {
                for (const auto &lineitem : lineitems)
                {
                    if (order.O_ORDERKEY == lineitem.L_ORDERKEY)
                    {
                        resultCount++;
                        // For simplicity, we just count the matches. In production, you might store results.
                        std::cout << "Customer Name: " << customer.C_NAME << ", Order Date: " << order.O_ORDERDATE << ", Quantity: " << lineitem.L_QUANTITY << std::endl;
                    }
                }
            }
        }
    }
}

int main()
{
    // Example usage of the data and calling the function (data initialization skipped)
    injectData(1500000, 150000, 1000);

    double threshold = 1000.0;
    int resultCount = 0;

    auto start = std::chrono::high_resolution_clock::now();
    processQueryCPU(customers, orders, lineitems, threshold, resultCount);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Total time taken by CPU: " << duration.count() << " ms\n";

    return 0;
}