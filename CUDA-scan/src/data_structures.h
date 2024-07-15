#pragma once
#include <vector>
#include <cstdio>
#include <iostream>
#include <cmath>

/**
 * 定义了数据结构和生成随机数据的函数;
 *
 */

// 定义各个表的列名
struct Customer
{
    int C_CUSTKEY;
    char C_NAME[100];
};

struct Orders
{
    int O_CUSTKEY;
    int O_ORDERKEY;
    char O_ORDERDATE[10];
    double O_TOTALPRICE;
};

struct Lineitem
{
    int L_ORDERKEY;
    int L_QUANTITY;
};

std::vector<Customer> customers;
std::vector<Orders> orders;
std::vector<Lineitem> lineitems;

template <typename T>
void generateRandomData(std::vector<T> &vec, size_t numElements);

template <>
void generateRandomData(std::vector<Customer> &vec, size_t numElements)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        Customer customer;
        customer.C_CUSTKEY = i;
        snprintf(customer.C_NAME, sizeof(customer.C_NAME), "Customer #%zu", i);
        vec.push_back(customer);
    }
    std::cout << "Generated " << numElements << " elements for Customers table" << std::endl;
}

template <>
void generateRandomData(std::vector<Orders> &vec, size_t numElements)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        Orders order;
        order.O_CUSTKEY = i;
        order.O_ORDERKEY = i;
        snprintf(order.O_ORDERDATE, sizeof(order.O_ORDERDATE), "2023-01-%02zu", (i % 30) + 1);
        if (i == 0)
        { // Ensure exactly one order has a total price > 1000
            order.O_TOTALPRICE = 1500.0;
        }
        else
        {
            order.O_TOTALPRICE = std::fmod(50.0 * i, 1000.0); // Keep other prices below 1000
        }
        vec.push_back(order);
    }
    std::cout << "Generated " << numElements << " elements for Orders table" << std::endl;
}

template <>
void generateRandomData(std::vector<Lineitem> &vec, size_t numElements)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        Lineitem item;
        item.L_ORDERKEY = i;
        item.L_QUANTITY = i % 5 + 1;
        vec.push_back(item);
    }
    std::cout << "Generated " << numElements << " elements for Lineitems table" << std::endl;
}

void injectData()
{

    // 假设 generateRandomData 是一个现有的函数，用来生成随机数据。
    generateRandomData(customers, 1500000);
    generateRandomData(orders, 15000000);
    generateRandomData(lineitems, 59986052);
}

// 可以指定各个表插入数据
void injectData(int numCust, int numOrder, int lineItmesNum)
{

    // 假设 generateRandomData 是一个现有的函数，用来生成随机数据。
    generateRandomData(customers, numCust);
    generateRandomData(orders, numOrder);
    generateRandomData(lineitems, lineItmesNum);
}