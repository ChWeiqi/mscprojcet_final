# Makefile

ifdef INCLUDEPATH
	INCLUDEPATH := $(INCLUDEPATH)
else
	INCLUDEPATH := /userhome/cs2/u3624833/anaconda3/envs/nvcc/include/
endif

# 编译器设置
CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++11 -I$(INCLUDEPATH)
NVCCFLAGS := -I$(INCLUDEPATH)

# 目标二进制文件目录
BIN_DIR := bin

# 默认目标
all: $(BIN_DIR)/CUDA-scan

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# 生成 CUDA 二进制文件
$(BIN_DIR)/CUDA-scan: src/sim_scan_2.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# 清理生成的二进制文件
clean:
	rm -rf $(BIN_DIR)/*
