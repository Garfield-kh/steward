# 可选的昇腾 NPU 后端

[English](README.md) | 中文

这是面向 toy 装配配置的一个新增可选 Python 包，对应提交 `07147e7764f5dce30838a9f3632d497f8067aeb8`。
它不修改原有的 Steward/GPU 源码、装配策略、技能、任务或成功判据。
使用原来的 Steward 入口不会开启 NPU；只有显式使用本包入口才开启。不包含模型权重。

## 组成

- `renderer/kernels.py`：盒子/圆柱求交、分块候选剔除、RGB 着色、深度、物体编号、立方体纹理采样。
- `renderer/dynamic_renderer.py`：MuJoCo 适配与工作进程管理。
- `renderer/render_worker.py`：在独立 Python 进程里执行 NPU 张量计算。
- `perception/sam3_api_srv.py`：可配置的 SAM 3 文本提示分割服务，支持 CUDA 或 NPU。
- `perception/graph_accel.py`、`decoder_meta.py`：可选的 NPU 图回放和主机侧元数据缓存，用开关显式启用。

物理和场景元数据仍在 CPU 上；像素求交、阴影、光照和纹理采样在 NPU 上执行。

## 安装和启用

验证环境：910B3 64 GiB、CANN 8.5.2、Python 3.11、torch/torch_npu 2.10.0，
模拟端 MuJoCo 3.17.0。分割端还需 SAM 3 和本地 `sam3.pt`。
先按机器实际环境准备 CANN、动态库、匹配的 PyTorch/NPU 依赖；本包不会自动升级这些。
SAM 需要四处兼容改动：固定官方版本与补丁见 `dependencies/README.md`，
补丁的前后文件哈希也已记录。不要直接用任意未修改的 SAM 安装代替。

在需要导入本包的 Python 环境中执行（依赖已准备好时）：

```sh
python -m pip install --no-deps -e /path/to/this/directory
```

模拟进程与 NPU 工作进程可以使用不同 Python 环境，两边均需安装本包或配置 `PYTHONPATH`。
模拟端需要原 Steward 和 MuJoCo；工作进程需要 NumPy、torch、torch_npu。

```sh
export STEWARD_ASSEMBLY_PROFILE=toy
export NPU_DEVICE=0
# 只有使用独立 NPU Python 环境时才需要：
export STEWARD_NPU_PYTHON=/path/to/npu/environment/bin/python
python -m steward_npu --help
```

把 `--help` 换成原来传给 `python -m steward.cli` 的参数即可。入口默认开启已测量的
A800 深度采样对齐（`STEWARD_NPU_MATCH_A800_DEPTH=1`）；仅支持 0/4 采样模式。
显卡编号是物理卡号，工作进程内部映射为逻辑 0 号。
可调参数：`STEWARD_NPU_RENDER_TILE=512`、`STEWARD_NPU_RENDER_THREADS=4`、
`STEWARD_NPU_RENDER_MEMORY_FRACTION=0.25`。显存比例是分配器上限，不是实测占用，
也不代表预留了整张卡。每个实验进程启动一个渲染工作进程，同进程相机共用。

启动分割服务：

```sh
export ASCEND_RT_VISIBLE_DEVICES=0
python -m steward_npu.perception.sam3_api_srv \
  --device-type npu --devices 0 --ckpt /path/to/sam3.pt \
  --host 127.0.0.1 --port 18230 --memory-fraction 0.45 \
  --npu-text-cache 16 --npu-empty-box-fastpath --npu-empty-geometry \
  --npu-shared-geometry --npu-batched-postprocess --npu-graphs
```

`--devices` 为可见设备中的逻辑编号；兼容旧参数名 `--gpus`。不加优化开关则走普通
推理路径。CUDA 模式用 `--device-type cuda`，不能带 NPU 专用开关。接口是 `/api/health` 与 `/api/segment`；具体请求格式、
缓存限制、日志参数见英文 README。服务没有鉴权，按原部署方式放在可信网络中。

## 速度与成绩

同场景、单路渲染耗时，包含场景更新及 RGB/深度/编号回传，单位秒/帧：

| 画面 | A800 SXM4 80GB | Kunpeng 920 CPU | 910B3 NPU |
| --- | ---: | ---: | ---: |
| 头部 2560×1920 | 0.2561 | 1.3095 | 0.6702 |
| 广角 480×360 | 0.00535 | 0.1410 | 0.1859 |
| 手部 848×480 | 0.01158 | 0.1600 | 0.2122 |

CPU 用 OSMesa/llvmpipe、8 线程；机器为四路 Kunpeng 920，共 192 核。
GPU 主机为双路 Xeon Platinum 8358P，共 64 核。
NPU 在头部大图上比 CPU 快，小图更慢；三种画面均比 A800 慢。

另一次一/二路并发测试中，NPU 单路延迟分别为头部 0.723/0.949 秒、广角
0.198/0.239 秒、手部 0.237/0.265 秒，总吞吐提高约 1.52/1.58/1.75 倍。
对应 CPU 延迟为 1.328/1.301、0.171/0.155、0.191/0.168 秒。
没有同条件 GPU 双路结果。原始汇总 CSV 在 `validation/`。

**装配策略不变的首批 8 局，四卡、每卡两路、每局最多 180 分钟：20 毫米判据 1 局成功，
6 毫米判据 0 局成功。** 成功局约 81.7 分钟首次达到四颗，131.1 分钟结束，
最大横向误差 16.55 毫米。32 局的批次正在进行。

渲染仍有限制：仅针对不透明盒子/圆柱、透视相机、方向光与 RGB 立方体纹理。
不支持通用网格、透明材质、完整 OpenGL 阴影/抗锯齿等效果。NPU 与 GPU/CPU 的图像
不是逐像素一致，边缘可能有很大的深度误差。

## 验证

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

检查范围和在 NPU 机器上的回归结果见 `validation/README.md`。
