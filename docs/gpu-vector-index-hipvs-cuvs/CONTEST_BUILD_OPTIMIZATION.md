# lib=cuvs 索引构建加速：实现说明、测试方法与性能报告

基线：`amd-cagra-contest-baseline-v4`（`ad3253a5`，`amd-cagra-contest-baseline-v1` 的后代）。
对外 SQL、FP32 数据语义、CAGRA 图度数（graph_degree=128、intermediate_graph_degree=256）、
检索参数（itopk/search_width）与 bridge ABI 均不变；seekdb C++ 代码与基线一致，`lib=vsag` 路径不受影响。

## 1. 改动

`tools/hipvs_bridge/seekdb_cuvs_bridge.c`：CAGRA 建图算法由 `NN_DESCENT`（100 轮）改为
`ITERATIVE_CAGRA_SEARCH`（在逐步扩大的 CAGRA 图上迭代检索得到 kNN 图，再按同样的度数剪枝）。

动机：在 W7900（gfx1100）上实测，NN-Descent 的迭代轮数决定了 GPU 建图耗时（100 轮 25.4s、
20 轮 7.4s），但减少轮数会使 Recall@100 下降超过 0.005 的门槛；迭代检索法在相同度数下更快且召回更高。

## 2. 实测结果（AMD Radeon Cloud，W7900 gfx1100，ROCm 7.2.4，容器 16 核 / 55GB）

### 2.1 CAGRA 建图本身（bridge 相同的 C API 与检索参数，100 条查询，暴力真值）

| 数据 | 建图算法 | 建图时间 | Recall@10 | Recall@100 |
|---|---|---|---|---|
| 100k × 128 | NN-Descent 100 轮（基线） | 25.4 s | 0.999 | 0.9969 |
| 100k × 128 | NN-Descent 20 轮 | 7.4 s | 0.995 | 0.9803 ✗ |
| 100k × 128 | NN-Descent 40 轮 | 13.5 s | 0.996 | 0.9931 |
| 100k × 128 | IVF-PQ | 3.7 s | 0.933 | 0.8824 ✗ |
| 100k × 128 | **迭代检索（本方案）** | **17.7–18.2 s** | **1.000** | **0.9979–0.9981** |
| 100k × 256 | NN-Descent 100 轮（基线） | 27.0 s | 0.996 | 0.9854 |
| 100k × 256 | **迭代检索（本方案）** | **23.9 s** | **1.000** | **0.9917** |

### 2.2 端到端：`CREATE VECTOR INDEX ... lib=cuvs` + 第一条查询（100k × 128，每版本 2 轮）

`vector_index_memory_saving_mode=false`（快照索引常驻内存，查询走 GPU CAGRA）：

| 版本 | 构建耗时 | Recall@10 | Recall@100 | 单查 P50 |
|---|---|---|---|---|
| 冻结基线 v4 | 67.97 / 65.46 s（中位 66.7） | 0.997 | 0.9956 | 23.3 ms |
| 本版本 | 58.01 / 57.91 s（中位 58.0） | **1.000** | **0.9986** | 23.5 ms |

构建加速比 **1.15×**；Recall 提升，单查延迟持平。

默认 `vector_index_memory_saving_mode=true` 时，DDL 结束后内存快照被释放，首查从快照表反序列化
VSAG 索引并走 CPU，基线与本版本都不调用 GPU（基线 38.3 / 38.6 s，Recall@10 0.67，进程未打开 GPU 设备）。

### 2.3 批量查询：`dbms_vector.batch_knn`（100k 底库，1000 条探针，Top-10）

| 版本 | 第 1 次 | 第 2 次 | 结果 |
|---|---|---|---|
| 基线 bridge | 28.46 s | 25.60 s | 10000 行，1000/1000 自身排第 1 |
| 本版本 bridge | 19.71 s | 17.79 s | 同上 |

加速 **1.44×**（batch_knn 每次调用都会重建 CAGRA）。

### 2.4 正确性

`validation/run_gpu_smoke.sh` 全部 12 项通过：L2 走 GPU、VSAG 隔离、非 L2 DDL 拒绝、新鲜度回退与重建、
过滤与删除回退、batch_knn Top-10 / Top-100（Recall 1.0000）、GPU 设备句柄、隐藏 GPU 回退、清理。

## 3. 试过但放弃的方案（实测无收益）

* 延迟 + 多线程并行插入 VSAG HNSW：100k 行 15 线程仍需 35 s，与单线程相当（该版本 VSAG 插入实际串行化），
  且阻塞了后台调度线程，导致 DDL 结束时快照交接出错（首查 0 行）。已撤回。
* DDL 序列化时提前在后台启动 CAGRA 建图：序列化本身很短，重叠收益可忽略（68.7 s vs 66.7 s）；
  在省内存模式下还会白跑一次 GPU 建图并拖慢查询。已撤回。

## 4. 复现方法

```bash
# 1) 编译 bridge 与 seekdb（容器限 16 核时不要用超过 -j16，否则会因内存超限被回收）
tools/hipvs_bridge/build.sh /workspace/bridge/libseekdb_cuvs_bridge.so
./build.sh release -DOB_BUILD_CUVS=ON -DOB_BUILD_CUVS_TRACE=ON \
  -DCUVS_BRIDGE_LIB=/workspace/bridge/libseekdb_cuvs_bridge.so --make -j14
# 2) 正确性
PATH=/opt/venv/bin:$PATH docs/gpu-vector-index-hipvs-cuvs/validation/run_gpu_smoke.sh \
  --observer build_release/src/observer/seekdb --bridge /workspace/bridge/libseekdb_cuvs_bridge.so \
  --base-dir /workspace/smoke --port 2981 --render /dev/dri/renderD134 --evidence /workspace/smoke_ev
# 3) 构建性能（observer 已启动、端口 2881；首轮去掉 --skip-load 以导入数据）
mysql -h127.0.0.1 -P2881 -uroot -e "alter system set vector_index_memory_saving_mode=false"
/opt/venv/bin/python3 docs/gpu-vector-index-hipvs-cuvs/validation/bench_cagra_build.py \
  --port 2881 --lib cuvs --n 100000 --dim 128 --nq 100 --rounds 2 --out result.json
```

基线与本版本使用同一 observer 数据目录交替启动，bridge 分别为 v4 源码与本版本源码编译的
`libseekdb_cuvs_bridge.so`。正式成绩以组织方云端评测为准。
