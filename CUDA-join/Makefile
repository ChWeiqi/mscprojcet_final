# Compiler settings
ifdef NVCC
	NVCC := $(NVCC)
else
        NVCC := /userhome/2072/msp23007/anaconda3/pkgs/cuda-toolkit/bin/nvcc
endif
ifdef GCC
	GCC := $(GCC)
else
        GCC := /userhome/2072/msp23007/anaconda3/bin/gcc
endif
ifdef CXX
	CXX := $(CXX)
else
        CXX := /userhome/2072/msp23007/anaconda3/bin/g++
endif
CFLAGS := -O3 -march=native -w
NVCCFLAGS := -O3 -arch=sm_61 -allow-unsupported-compiler -std=c++17 -Xcompiler -fPIC -ccbin=$(GCC) -w

# CUDA paths
ifdef CUDA_PATH
	CUDA_PATH := $(CUDA_PATH)
else
        CUDA_PATH := /userhome/2072/msp23007/anaconda3/pkgs/cuda-toolkit
endif
CUDA_INCLUDE := $(CUDA_PATH)/include

# Project paths
PROJECT_INCLUDE := $(shell pwd)

# Source files
CU_SRCS := $(wildcard *.cu)
C_SRCS := $(wildcard *.c)

# Object files
OBJ_DIR := obj
CU_OBJS := $(addprefix $(OBJ_DIR)/,$(notdir $(CU_SRCS:.cu=.o)))
C_OBJS := $(addprefix $(OBJ_DIR)/,$(notdir $(C_SRCS:.c=.o)))

# Target executable
TARGET := cuda-join
BIN_DIR := bin

# Default target
all: $(BIN_DIR)/$(TARGET)

# Create obj and bin directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Linking
$(BIN_DIR)/$(TARGET): $(CU_OBJS) $(C_OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# CUDA compilation
$(OBJ_DIR)/%.o: %.cu | $(OBJ_DIR)
	@echo "CUDA_INCLUDE: $(CUDA_INCLUDE)"
	@echo "PROJECT_INCLUDE: $(PROJECT_INCLUDE)"
	@ls -l $(CUDA_INCLUDE)/crt || echo "crt directory not found in CUDA_INCLUDE"
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INCLUDE) -I$(PROJECT_INCLUDE) -c $< -o $@

# C compilation
$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(GCC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
