#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

#include "utils.cuh"
#include "experiment_util.cuh"

/*
定义宏:
1. 用于获取指针的基础类型
2. 用于定义指针类型
*/
#define GET_DATA_TYPE(p) using p##_t = std::remove_pointer_t<decltype(p)>;
#define DEF_DATA_TYPE(p, type) using p##_t = type;

// 数据文件的前缀
#define TPC_DATA_PREFIX "../tpc-data-files"

/*
接连三个定义宏:
1. LOAD_COL 用于加载列数据
*/
#define LOAD_COL(p, N) \
    long *p;           \
    GET_DATA_TYPE(p)   \
    alloc_load_tpch_column(TPC_DATA_PREFIX "/" #p ".bin", p, (N));

// 2. READ_COL_SHUFFLE: 用于读取并且打乱列数据
#define READ_COL_SHUFFLE(p, N, from, to, seed) \
    to *p;                                     \
    GET_DATA_TYPE(p)                           \
    printf("Read Column: %p\n", (void *)p);    \
    read_col<from, to>(TPC_DATA_PREFIX "/" #p ".bin", p, (N), true, (seed));

// 3. RET_COL: 用于获取列数据指针
#define RET_COL(m) auto c##m = t.template get_typed_ptr<m>();

// 模板函数，用于读取列数据
template <class T>
void read_col(const std::string &file_name, T *&dst, const int N, bool shuffle = false, const int seed = 42)
{
    dst = new T[N];

    // 如果 shuffle 设置为 true; 则尝试从 file_name.shuffled 内读取数据
    auto shuffle_path = file_name + ".shuffled";
    printf("Enter tpc_utils.hpp read_col() function, file path: %s\n", shuffle_path.c_str());

    if (shuffle)
    {
        std::ifstream file(shuffle_path, std::ios::binary);

        if (!file.is_open())
        {
            printf("Error: Failed to open file %s\n", shuffle_path.c_str());
            perror("Error details");
        }
        else
        {
            // printf("File opened successfully\n");

            if (!file.good())
            {
                printf("Error: File stream is not in a good state\n");
                printf("EOF: %d, fail: %d, bad: %d\n",
                       file.eof(), file.fail(), file.bad());
            }
            else
            {
                // printf("File stream is in a good state\n");

                file.read(reinterpret_cast<char *>(dst), N * sizeof(T));
                if (file.gcount() != N * sizeof(T))
                {
                    printf("Warning: Read %ld bytes, expected %ld bytes\n", file.gcount(), N * sizeof(T));
                }
                else
                {
                    // printf("Read correct number of bytes\n");
                }

                file.close();
                printf("Finished reading shuffled file. First element: %ld\n", static_cast<long>(dst[0]));
                return;
            }
        }
    }

    alloc_load_column(file_name, dst, N);

    // 如果 .shuffled 文件不存在，则会尝试从原始文件中加载数据，并且将其写入 .shuffled 文件内
    if (!shuffle)
        return;

    std::default_random_engine e(seed);
    std::shuffle(dst, dst + N, e);

    write_binary_file(dst, N, shuffle_path);
}

// 模板函数，用于读取并转换列数据类型
template <class T, class U>
void read_col(const std::string &file_name, U *&dst, const int N, bool shuffle = false, const int seed = 42)
{
    T *temp;
    read_col(file_name, temp, N, shuffle, seed);

    if constexpr (std::is_same<T, U>::value)
    {
        dst = temp;
        return;
    }

    dst = new U[N];
    for (int i = 0; i < N; i++)
    {
        dst[i] = static_cast<U>(temp[i]);
    }
    delete[] temp;
}