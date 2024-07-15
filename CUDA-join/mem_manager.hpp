#include <unordered_map>
#include <iostream>
#include <cuda.h>
#include <list>
#include <stdexcept>
#include <utils.cuh>

template <size_t kInitialPoolSize, size_t kAlignBits = 256UL>
class UserSpaceMM
{
private:
    using block_type = struct
    {
        char *data{nullptr};
        size_t sz{0};
    };

public:
    UserSpaceMM()
    {
        std::cout << "Memory pool has initial size " << kInitialPoolSize << " bytes.\n";
        cudaMalloc(&root, kInitialPoolSize);
        if (!root)
        {
            throw std::runtime_error("Initializing memory pool fails.\n");
        }

        // block_type root_block = {.data = (char *)root, .sz = kInitialPoolSize};
        block_type root_block{(char *)root, kInitialPoolSize};
        free_list.insert(free_list.cend(), root_block);
    }

    void allocate(void **ptr, bool clear, size_t sz, cudaStream_t stream)
    {
        std::cout << "Entering allocate function. Requested size: " << sz << " bytes" << std::endl;

        if (sz == 0)
        {
            std::cout << "!!WARNING!! Allocation of 0 bytes!!!\n";
            *ptr = nullptr;
            return;
        }

        auto b = find_block(sz);
        *ptr = (void *)(roundup(b.data));

        std::cout << "Block found. Address: " << *ptr << ", Size: " << b.sz << " bytes" << std::endl;

        if (clear)
        {
            cudaError_t err = cudaMemsetAsync(*ptr, 0, sz, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "cudaMemsetAsync failed: " << cudaGetErrorString(err) << std::endl;
                // 处理错误...
            }
        }

        alloc_map[*ptr] = b;
        mem_used += b.sz;
        if (mem_used > peak_mem_used)
            peak_mem_used = mem_used;

        std::cout << "Memory allocated. Current usage: " << mem_used << " bytes, Peak usage: " << peak_mem_used << " bytes" << std::endl;

        // 验证内存可写入
        int testValue = 42;
        cudaError_t err = cudaMemcpyAsync(*ptr, &testValue, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to write to allocated memory: " << cudaGetErrorString(err) << std::endl;
            // 处理错误...
        }

        // 验证内存可读取
        int readValue = 0;
        err = cudaMemcpyAsync(&readValue, *ptr, sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to read from allocated memory: " << cudaGetErrorString(err) << std::endl;
            // 处理错误...
        }
        else if (readValue != testValue)
        {
            std::cerr << "Memory verification failed. Expected " << testValue << ", got " << readValue << std::endl;
            // 处理错误...
        }
        else
        {
            // std::cout << "Memory verification successful." << std::endl;
        }

        // std::cout << "Exiting allocate function." << std::endl;
    }

    void release(void *ptr, cudaStream_t stream)
    {
        if (auto search = alloc_map.find(ptr); search != alloc_map.end())
        {
            insert_block(search->second);
            alloc_map.erase(search);
            mem_used -= search->second.sz;
        }
        else
        {
            std::cout << "!!WARNING!! Potential double-free: "
                      << ptr << " is not allocated through MemoryManager.\n";
        }
    }

    size_t get_peak_mem_used() const
    {
        return peak_mem_used;
    }

private:
    // find a block in the free list that is at least as large as sz
    block_type find_block(const size_t sz)
    {
        block_type ret;

        for (auto b = free_list.begin(); b != free_list.end(); b++)
        {
            if (can_alloc(*b, sz))
            {
                char *alloc_start = roundup(b->data);

                ret.data = b->data;
                ret.sz = sz + alloc_start - b->data;

                if (b->sz == ret.sz)
                {
                    free_list.erase(b);
                }
                else
                {
                    b->data += ret.sz;
                    b->sz -= ret.sz;
                }

                return ret;
            }
        }

        int i = 0;
        for (auto b = free_list.begin(); b != free_list.end(); b++, i++)
        {
            std::cout << "Block " << i << ": " << b->sz << " bytes\n";
        }

        throw std::bad_alloc();
    }

    // Insert the block back to the free list, merge if necessary
    void insert_block(const block_type &free)
    {
        if (free_list.empty())
        {
            free_list.insert(free_list.cend(), free);
            return;
        }

        auto const next = std::find_if(free_list.begin(), free_list.end(), [free](block_type const &blk)
                                       { return free.data < blk.data; });
        auto const prev = (next == free_list.cbegin()) ? next : std::prev(next);

        // Coalesce with neighboring blocks or insert the new block if it can't be coalesced
        bool const merge_prev = (prev->data + prev->sz == free.data);
        bool const merge_next = (next != free_list.cend()) && (free.data + free.sz == next->data);

        if (merge_prev && merge_next)
        {
            prev->sz += free.sz + next->sz;
            free_list.erase(next);
        }
        else if (merge_prev)
        {
            prev->sz += free.sz;
        }
        else if (merge_next)
        {
            next->data = free.data;
            next->sz += free.sz;
        }
        else
        {
            free_list.insert(next, free); // cannot be coalesced, just insert
        }
    }

    size_t roundup(const size_t value)
    {
        return (value + (kAlignBits - 1)) & ~(kAlignBits - 1);
    }

    char *roundup(const char *ptr)
    {
        uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
        value = (value + (kAlignBits - 1)) & ~(kAlignBits - 1);
        return (char *)value;
    }

    bool can_alloc(const block_type &block, const size_t sz)
    {
        auto ptr = roundup(block.data);
        return ptr + sz < block.data + block.sz;
    }

private:
    std::unordered_map<void *, block_type> alloc_map;
    std::list<block_type> free_list;
    void *root{nullptr};
    size_t mem_used{0};
    size_t peak_mem_used{0};
};