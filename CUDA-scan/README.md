# CUDA-Scan

We conducted 3 preliminary experiments on  this demo to verify our assumptions and get some preliminary conclusions.



| Experiment Group No | Experiment Details                                           | Files                                                       |
| ------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| Group 1             | Concentrate on the deployment of the Scan Operator  on both GPU and CPU platforms | sim_scan_2.cu <br />sim_scan_2.cpp                          |
| Group 2             | Expand on the groundwork laid by Group 1 Incorporated the Scan, Selection, and Projection  Operators. Add some calculation. | scan_agg.cu<br />scan_agg.cpp                               |
| Group 3             | Add the Join Operator to our experimental design.            | scan_select_project_gpu.cu<br />scan_select_project_gpu.cpp |



## Directory

- src: contain all the source code files
- target: contain all the compile results from the source code



## Files Description

- `sim_scan_2.cu`: Scans three data tables using CUDA.
- `sim_scan_2.cpp`: Scans three data tables based on CPU.
- `scan_agg.cu`: Scans three tables, joins them, and performs computations using CUDA.
- `scan_agg.cpp`: Scans three tables, joins them, and performs computations based on CPU.
- `scan_select_project_gpu`.cu: Scan, Selection, Projection and Join by operating three tables using CUDA
- `scan_select_project_cpu`.cpp: Scan, Selection, Projection and Join by operating three tables based on CPU
- `utils.cuh`: CUDA utility functions.
- `data_structures.h`: Defines the column names for the three tables. Generate data utilities functions.

## Compilation and Execution

### Steps to Compile

1. Open a command line interface.
2. Navigate to the directory containing the CUDA code files, such as `example.cu`.
3. Use the `nvcc` command to compile the code and specify the output file name. Typically, the output can be an object file (ending in `.o`) or an executable file. For instance, to generate an executable named `example`, use the following command:

    ```bash
    nvcc -o example example.cu
    ```

    Alternatively, to generate an object file:

    ```bash
    nvcc -c -o example.o example.cu
    ```

4. If the compilation is successful, you will see an executable named `example` or an object file named `example.o` in the current directory.
5. To run the executable on a Linux system, use the command:

    ```bash
    ./example
    ```

### Notes on `nvcc`

The `nvcc` compiler will automatically handle the compilation of CUDA code and invoke `gcc` (or `cl.exe` on Windows) to compile the main program and handle non-CUDA code. You might need to specify additional compilation options, such as:

- `-arch`: Specifies the compute capability to ensure the generated code runs on a specific CUDA architecture.
- `-G`: Enables debugging.
- `-lineinfo`: Retains source line information.
- `-O2` or `-O3`: Specifies the compiler optimization level.
- `-use_fast_math`: Uses fast math library.

For example, if your GPU supports compute capability 5.0, you can compile the code with:

```bash
nvcc -arch=sm_50 -o example example.cu
```

### Compiling for This Demo

To compile the demo with specific options, use the following command. [For example, compile scan_agg.cu]:

Enter the src directory and execute

```
nvcc -arch=sm_89 -o scan_agg_gpu scan_agg.cu -O3
```

Then run the executable with:

```
./scan_agg_gpu
```
