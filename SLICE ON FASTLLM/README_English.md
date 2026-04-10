# SLICE

## Introduction

​	SLICE is an innovative scheduling scheme designed for edge computing scenarios with differentiated SLO requirements. By combining a utility-maximizing request scheduling algorithm with a dynamic iterative control mechanism for generation rates, SLICE significantly enhances SLO attainment for LLM inference services.

​	The core concept of SLICE revolves around ”real-time task prioritization and on- demand resource allocation”. Specifically, SLICE gives prior- ity to scheduling real-time tasks to ensure they meet their dead- lines. Additionally, it allocates decoding rates in accordance with SLO requirements. This approach prevents low-demand tasks from excessively consuming computational resources, thereby enabling more concurrent tasks to run even with limited computing power . By adhering to this principle, SLICE achieves a synergistic improvement in both SLO attainment and the successful completion of real- time tasks within their deadlines. To implement this scheduling concept, SLICE employs a two-stage scheduling strategy.

​	In the first phase (i.e., the task selection phase), each newly arrived task is assigned a utility value that reflects its impor- tance. Notably, real-time tasks are assigned significantly higher utility values than non-real-time tasks (typically 10 to 100 times higher) in the edge scenarios. By maximizing the total utility value, the system selects a batch of tasks for decoding, ensuring real-time task prioritization while maximizing non- real-time task throughput. In the second phase (i.e., the rate allocation phase), the decoding rate is dynamically adjusted based on the SLO requirements of each task. To minimize system overhead, this paper proposes an innovative decode-mask matrix that enables cycle-based scheduling to achieve rate control for individual tasks. Specifically, the number of columns in the matrix represents the total number of decoding operations per cycle, while each row corresponds to one scheduled task, recording its decoding schedule within the cycle. In the matrix, setting the nth column of a row to ”1” indicates that the corresponding task participates in the nth decoding operation of that cycle. For tasks with stricter TPOT SLO requirements (i.e., those requiring a higher decoding rate), the ”1”s are distributed more densely across their respective rows, thereby enabling a higher decoding rate. During the actual scheduling process, the system performs column-wise scanning starting from the first column in each cycle, triggering a decoding operation upon completing the scan of each column. For every scan, the system groups all tasks corresponding to rows marked ”1” in the current column into a single decoding batch. Through this iterative scanning and dynamic batch aggregation mechanism, the system can dynamically adjust the decoding rate of each task according to its SLO requirements.

## Quick Start

```bash
git clone https://github.com/willzhoupan/docker-mininet

cd slice
mkdir build
cd build
cmake .. -DUSE_CUDA=ON
make -j
```

## User Guide

### 1. How to Start a Model

The basic command format for starting a model is as follows:

```
./benchmark -p chatglm-6b-int4.flm --slice --test-mode --poisson
```

Here, `model` can be:

- A local model path, e.g., `/mnt/ChatGLM-6B`. For high-speed model downloads, refer to [Model Download](#Model Download)

Local models currently support the following formats:

- Original models in `FP16` or `FP8` format, e.g., `ChatGLM/ChatGLM-6B-FP16`
- Models in `Int4` format, e.g., `ChatGLM/ChatGLM-6B-Int4`
- Models in `Fastllm` format, e.g., `fastllm/DeepSeek-V3-0324-INT4`. You can also download the original model and export it using the commands in [Model Export](#Model Export).

### 2. How to Set Runtime Parameters

Runtime parameters can be set via the following options.

Note: Performance is not always directly proportional to parameter settings. For high-performance requirements, experimentation in different directions is recommended.

- `-t` or `--threads`:
  - **Description**: Sets the number of CPU threads to use.
    - When `device` is `cpu`, this parameter determines the number of threads used for inference.
    - When `device` is `numa`, the number of inference threads is primarily determined by the environment variable `FASTLLM_NUMA_THREADS`; the `threads` parameter should be set lower (recommended value is 1).

Performance varies significantly across different hardware with different parameters. Generally, on CPUs, it is not recommended to exceed the number of physical cores with the thread count.

- `--slice`:
  - **Description**: Enables the SLICE scheduler.
- `--help`:
  - **Description**: Views detailed information about module parameters.
- `--test-mode`:
  - **Description**: Enables SLICE test mode.
- `--concurrent`:
  - **Description**: Uses concurrent mode; sets the number of concurrent tasks.
- `--poisson`:
  - **Description**: Enables the Poisson distribution task arrival mode.
- `--ratios`:
  - **Description**: Hyperparameter coefficients.

## Task Example

Example:

```
 {
         "Describe your ideal vacation. Include the destination, one activity you would do, and why you chose it. Keep the description within 100 words.",
            false, false, 1, {}, {}, {}, {}, 0.04363281, 0, 0, 0, -1, false, 1, 120
 },
```

## Model Acquisition

### Model Download

Use the following command to download the model to your local machine (uses high-speed mirrors, no scientific internet access required):

```
ftllm download ChatGLM/ChatGLM-6B
```

### Model Export

If using quantized models (e.g., `--dtype int4`), the model is quantized on-the-fly during loading each time, which can be slow.

`ftllm export` is a tool for exporting and converting model weights. It supports converting model weights to different data types. Below are detailed instructions on how to use `ftllm export`.

#### Command Format

``` sh
ftllm export <model_path> -o <output_path> --dtype <data_type> -t <thread_count>
```

#### Example Command

``` sh
ftllm export /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-INT4 --dtype int4 -t 16
```

#### Mixed Precision

Mixed precision can be achieved by specifying `--moe_dtype`, for example:

``` sh
ftllm export /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-FP16INT4 --dtype float16 --moe_dtype int4 -t 16
```

#### Loading the Exported Model

The usage of the exported model is similar to the original model. The `--dtype` parameter is ignored when using the exported model.

For example:

``` sh
ftllm run /mnt/DeepSeek-V3-INT4/
```



