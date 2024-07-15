# contrib/gpu_executor/Makefile
# include CUDA-scan/Makefile

MODULE_big = gpu_accelerator
EXTENSION = gpu_accelerator
OBJS = main.o gpu_scan.o gpu_join.o chunk_executor.o gpu_accelerator_func.o gpu_cost.o
HEADERS = gpu_accelerator.h
DATA = gpu_accelerator--1.0.sql

ifdef PG_CONFIG:
	PG_CONFIG := $(PG_CONFIG)
else
        PG_CONFIG := /userhome/2072/msp23007/pgsql/bin/pg_config
endif
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
