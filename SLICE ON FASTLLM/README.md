# SLICE ON FASTLLM

## 快速启动

```bash
git clone https://github.com/willzhoupan/SLICE

cd slice
mkdir build
cd build
cmake .. -DUSE_CUDA=ON
make -j
```

## 使用指南

### 1. 如何启动模型

基本的启动命令格式如下：

```
./benchmark -p chatglm-6b-int4.flm --slice --test-mode --poisson
```

这里的`model`可以是:

- 本地模型路径。例如`/mnt/ChatGLM-6B`，高速下载模型可以参考 [模型下载](#模型下载)

本地模型，目前支持以下几种格式:

- `FP16`, `FP8`格式的原始模型，例如`ChatGLM/ChatGLM-6B-FP16`
- `Int4`格式的模型，例如`ChatGLM/ChatGLM-6B-Int4`
- `Fastllm`格式的模型，例如`fastllm/DeepSeek-V3-0324-INT4`。也可以下载原始模型后通过 [模型导出](#模型导出) 中的命令导出

### 2. 如何设定运行参数

可以通过下列参数设置运行参数。

需要注意的是，速度和参数设置并不一定正相关，如果对性能要求高，可以多方向尝试一下

- `-t` 或 `--threads`:
  - **描述**: 设置使用的CPU线程数。
    - 当`device`为`cpu`时，这个参数决定了推理使用的线程数
    - 当`device`为`numa`时，推理线程数主要由环境变量`FASTLLM_NUMA_THREADS`决定，`threads`参数请设得小一点（推荐设为1）

不同硬件上，不同参数发挥出的性能有很大不同。一般而言，CPU上使用的线程数不建议超过物理核数

- --slice:
- **描述**: 启用SLICE调度器。
- `--help`:
  - **描述**: 查看模块参数详细信息。
- --test-mode:
  - **描述**: 启用SLICE测试模式。
- --concurrrent :
  - **描述**: 采用并发模式 设置并发任务数。
- --poisson:
  - **描述**: 启用泊松分布的任务到达模式
- `--ratios`:
  - **描述**: 超参系数。

## 任务示例

示例：

```
 {
         "Describe your ideal vacation. Include the destination, one activity you would do, and why you chose it. Keep the description within 100 words.",
            false, false, 1, {}, {}, {}, {}, 0.04363281, 0, 0, 0, -1, false, 1, 120
 },
```

## 模型获取

### 模型下载

可以使用如下命令将模型下载到本地（使用高速镜像，无需科学上网）

```
ftllm download ChatGLM/ChatGLM-6B
```


### 模型导出

如果使用量化加载模型（如`--dtype int4`），那么每次读取模型时会在线量化，读取速度较慢。

ftllm export 是一个用于导出和转换模型权重的工具。它支持将模型权重转换为不同的数据类型。以下是如何使用 ftllm export 的详细说明。

#### 命令格式

``` sh
ftllm export <模型路径> -o <输出路径> --dtype <数据类型> -t <线程数>
```

#### 示例命令

``` sh
ftllm export /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-INT4 --dtype int4 -t 16
```

#### 混合精度

可以通过指定`--moe_dtype`来实现混合精度，例如

``` sh
ftllm export /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-FP16INT4 --dtype float16 --moe_dtype int4 -t 16
```

#### 加载导出后的模型

导出后的模型使用方法和原始模型类似，使用导出模型时`--dtype`参数将被忽略

例如

``` sh
ftllm run /mnt/DeepSeek-V3-INT4/
```
