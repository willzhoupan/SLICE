# SLICE

## 介绍

​	SLICE一种针对具有差异化SLO需求的边缘计算场景而设计的创新调度方案。通过将效用最大化请求调度算法与生成速率的动态迭代控制机制相结合，SLICE显著提高了LLM推理服务的SLO实现。

​	SLICE坚持按需分配原则，在保证实时任务的截止时间约束的同时，最大限度地实现SLO。具体来说，SLICE通过动态效用率量化和令牌掩码矩阵实现了一种两阶段调度机制。在第一阶段（任务选择）中，每个传入任务被分配一个效用值，实时任务比非实时任务接收的效用值要高得多（通常是10-100倍）。通过最大化总效用值，系统选择一批任务进行解码，在保证实时任务优先级的同时最大化非实时任务吞吐量。第二阶段（速率分配）根据每个任务的SLO需求动态调整解码速率。为了最小化系统开销，SLICE使用一种新颖的令牌掩码矩阵用于速率控制。具体来说，矩阵的列表示每个周期的解码操作总数，以行严格对应于计划任务，记录每个任务的解码时间。系统从每个周期的第一列开始逐列扫描，在扫描每一列时触发解码操作——当前列中标记为“1”的行中的所有任务构成解码批。通过这种迭代扫描和批处理过程，系统动态调整任务解码频率以满足SLO要求。

## 快速启动

```bash
git clone https://github.com/willzhoupan/docker-mininet

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
