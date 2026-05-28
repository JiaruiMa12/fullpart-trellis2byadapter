# FullPart Adapter to TRELLIS2 生成流程报告

## 1. 目标概述

本流程的目标是将 FullPart 生成的全局稀疏 latent 表示桥接到 TRELLIS2 的 shape latent 空间，并进一步使用 TRELLIS2 的纹理生成与解码模块导出带纹理三维模型。

当前报告重点区分两条链路。

推荐用于质量验证的链路为：

```text
adapter 训练数据 old_slat
  -> FullPartToTrellis2SLatAdapter
  -> TRELLIS2 shape_slat 初值
  -> 可选 TRELLIS2 shape_slat_flow_model refinement
  -> TRELLIS2 texture latent sampling
  -> TRELLIS2 decoder
  -> textured GLB
```

实验性完整生成链路为：

```text
输入图像
  -> TRELLIS2 sparse structure sampler
  -> 随机或零初始化 FullPart old global_slat
  -> FullPartToTrellis2SLatAdapter
  -> TRELLIS2 shape_slat 初值
  -> TRELLIS2 shape_slat_flow_model refinement
  -> TRELLIS2 texture latent sampling
  -> TRELLIS2 decoder
  -> textured GLB
```

当前已经验证：

- 端到端脚本可以完整运行并导出 GLB。
- adapter 在训练集 seen sample 上可以预测与 GT TRELLIS2 latent 接近的 latent。
- 使用训练集中见过的 old_slat + GT coords 时，桥接路径具备可行性。
- 当前报告流程不依赖 FullPart Stage2，不加载 `s2.ckpt`。
- 随机初始化 old_slat 虽然可以端到端运行，但质量很低，因为它明显不在 adapter 的训练输入分布内。
- 当前更合理的质量验证方式是使用 adapter 训练数据中的真实 old_slat 生成。

---

## 2. 代码与模型路径

### 2.1 主代码仓库

```text
/root/data/fullpart-main
```

### 2.2 TRELLIS2 代码仓库

```text
/root/data/TRELLIS.2
```

### 2.3 FullPart checkpoints

当前初始化 latent 流程不使用 FullPart Stage1 / Stage2 checkpoint。

旧版完整 FullPart pipeline 中曾使用的 checkpoint 如下，仅作为可选历史路径记录：

```text
Stage1: /root/data/pretrained_models/fullpart/ckpts/pytorch_model-001.ckpt.download/s1.ckpt
Stage2: /root/data/pretrained_models/fullpart/ckpts/pytorch_model-001.ckpt.download/s2.ckpt
```

当前报告流程明确不加载：

```text
/root/data/pretrained_models/fullpart/ckpts/pytorch_model-001.ckpt.download/s2.ckpt
```

### 2.4 Adapter checkpoint

```text
/root/data/fullpart-main/exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt
```

### 2.5 TRELLIS2 pretrained checkpoint

```text
/root/data/pretrained_models/trellis.2-4B/ckpts
```

### 2.6 TRELLIS2 config

```text
/root/data/pretrained_models/trellis.2-4B/ckpts/pipeline_local.json
```

### 2.7 TRELLIS2 shape latent normalization stats

```text
/root/data/fullpart-main/dataset/partversexl/trellis2_shape_latents_stats.json
```

---

## 3. 运行环境

推荐使用 conda 环境：

```text
trellis2
```

Python 路径示例：

```text
/opt/miniconda3/envs/trellis2/bin/python
```

关键依赖与兼容性：

- `transformers==4.57.3`
- `diffusers` 当前环境版本
- CUDA + PyTorch
- `spconv`
- `flash_attn`
- `flex_gemm`
- TRELLIS2 依赖
- FullPart 依赖

关键环境变量：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPCONV_ALGO=native
```

说明：

- `transformers 4.57.3` 同时兼容 DINOv3 与当前 `diffusers`。
- `transformers 5.x` 会导致 `diffusers` 缺少 `FLAX_WEIGHTS_NAME`。
- 低版本 `transformers` 会导致 DINOv3 的 `dinov3_vit` 类型无法识别。

---

## 4. 阶段一：输入数据

### 4.1 输入图像

示例：

```text
assets/demo_examples/toy_gun.png
```

格式要求：

- RGB 或 RGBA 图像。
- 进入 TRELLIS2 前会统一转换为 RGBA。
- TRELLIS2 条件编码默认使用 512 分辨率。

### 4.2 Part bounding boxes

示例：

```text
assets/demo_examples/toy_gun.npy
```

日志中示例 shape：

```text
torch.Size([12, 2, 3])
```

含义：

- 12 个 part box。
- 每个 box 用两个三维点表示。
- 输入给 FullPart Stage1 / Stage2 用于 part-aware 生成。

---

## 5. 阶段二：FullPart Stage1

### 5.1 使用模型

FullPart Stage1 pipeline：

```text
src/pipelines/jointdit_single_3d_pipeline.py
```

Stage1 config：

```text
src.configs.train_configs.personal_configs_part
```

Stage1 checkpoint：

```text
/root/data/pretrained_models/fullpart/ckpts/pytorch_model-001.ckpt.download/s1.ckpt
```

### 5.2 输入

Stage1 输入包括：

- 条件图像 tensor。
- part bounding boxes。
- Stage1 config。
- Stage1 checkpoint。

### 5.3 输出

Stage1 生成 part-level 或 voxel-level 中间结果，保存到输出目录，例如：

```text
outputs/.../stage1/{sample_id}/
```

Stage2 会读取 Stage1 输出作为条件。

### 5.4 关键修复

已修复 Stage1 中 image condition dtype 与 transformer dtype 不一致的问题。

修复点：

```text
src/pipelines/jointdit_single_3d_pipeline.py
```

核心原因：

- transformer 可能运行在 bf16。
- image condition 默认是 fp32。
- 需要将 image condition 转为 transformer dtype。

---

## 6. 阶段三：初始化 FullPart global sparse latent

### 6.1 是否使用模型

实验性完整生成流程不使用 FullPart Stage2 模型，不加载：

```text
/root/data/pretrained_models/fullpart/ckpts/pytorch_model-001.ckpt.download/s2.ckpt
```

也不调用：

```text
src/pipelines/jointdit_single_3d_pipeline_stage2.py
```

### 6.2 随机初始化路径的输入

初始化 latent 的输入包括：

- TRELLIS2 sparse structure sampler 生成的 target coords。
- 初始化方式：`normal` 或 `zeros`。
- token 数量上限：`--init-global-slat-num-tokens`。

### 6.3 输出

输出一个 FullPart old latent 空间格式的 sparse latent：

```text
feats: [N, 8]
coords: [N, 4]
```

其中：

- `feats` 为初始化得到的 8-channel old latent。
- `coords` 复用 TRELLIS2 sparse coords，用于让 adapter 输出与 TRELLIS2 decoder 期望的坐标结构对齐。

### 6.4 当前实现

主脚本参数：

```text
--init-global-slat
--init-global-slat-mode normal
--init-global-slat-num-tokens 20000
```

真实运行日志示例：

```text
初始化 Global SLat 完成: feats torch.Size([N, 8]), coords torch.Size([N, 4])
```

### 6.5 global_slat 含义

`global_slat` 是 FullPart 旧 latent 空间中的 sparse latent：

- feature channel 数为 8。
- 坐标为 sparse grid 坐标。
- 它不是 TRELLIS2 decoder 可以直接解码的 latent。
- 必须通过 adapter 转为 TRELLIS2 shape latent。

### 6.6 质量问题说明

随机或零初始化的 `global_slat` 只保证张量形状正确：

```text
feats: [N, 8]
coords: [N, 4]
```

但它不保证落在 adapter 训练时见过的 old_slat 分布上。

因此虽然该路径已经可以端到端运行并导出：

```text
trellis2_shape_flow_textured.glb
```

但生成质量通常很低。

当前结论是：

```text
随机初始化 old_slat 适合验证 pipeline 连通性，不适合作为质量评估主路径。
```

### 6.7 推荐替代：使用 adapter 训练数据 old_slat

更合理的质量验证路径直接读取训练数据中的真实 old_slat：

```text
dataset/partversexl/textured_mesh_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/{asset_id}.npz
```

该数据与 adapter 训练时的输入分布一致，更适合验证：

```text
FullPart old latent -> TRELLIS2 latent
```

这个桥接映射本身是否有效。

---

## 7. 阶段四：Adapter 桥接

### 7.1 使用模型

Adapter 类：

```text
src/models/autoencoders/slat_adapter_trellis2.py
```

模型名称：

```text
FullPartToTrellis2SLatAdapter
```

checkpoint：

```text
/root/data/fullpart-main/exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt
```

### 7.2 Adapter 功能

Adapter 完成如下映射：

```text
FullPart global_slat feats: [N_old, 8]
FullPart global_slat coords: [N_old, 4]
Target TRELLIS2 coords: [N_t2, 4]
  -> TRELLIS2 shape_slat feats: [N_t2, 32]
```

即：

```text
old_slat + target_coords -> predicted TRELLIS2 shape_slat
```

### 7.3 输入

Adapter 输入为：

- FullPart global_slat features。
- FullPart global_slat coords。
- target TRELLIS2 sparse coords。

### 7.4 输出

Adapter 输出为 TRELLIS2 shape latent：

```text
feats: [N_t2, 32]
coords: [N_t2, 4]
```

示例日志：

```text
Adapter 转换完成: feats torch.Size([5016, 32]), coords torch.Size([5016, 4])
```

### 7.5 坐标来源

当前支持两种 target coords：

#### 7.5.1 remap coords

由 adapter 内部 `remap_coords` 从旧坐标映射得到。

问题：

- 容易与 TRELLIS2 真实 sparse structure 分布不一致。
- 生成质量较差。

#### 7.5.2 TRELLIS2 sparse sampler coords

使用 TRELLIS2 自己的 sparse structure sampler 从条件图像生成 coords。

流程：

```text
condition image
  -> TRELLIS2 image condition model
  -> sparse_structure_sampler
  -> target_coords
```

当前主 pipeline 默认使用：

```text
--adapter-target-coords trellis2_sparse
```

示例日志：

```text
TRELLIS2 target coords: torch.Size([5016, 4])
min tensor([0, 0, 25, 17])
max tensor([0, 63, 38, 46])
```

### 7.6 关键坐标修复

TRELLIS2 sparse coords 的 batch index 必须为 0。

已修复：

```python
trellis2_slat.coords[:, 0] = 0
```

否则会触发 attention batch size mismatch。

---

## 8. 阶段五：TRELLIS2 sparse structure sampling

### 8.1 使用模型

TRELLIS2 pipeline：

```text
/root/data/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py
```

Pipeline 类：

```text
Trellis2ImageTo3DPipeline
```

### 8.2 条件图像编码模型

在 `pipeline_local.json` 中配置：

```text
DinoV3FeatureExtractor
```

DINOv3 checkpoint：

```text
/root/data/pretrained_models/dinov3-vitl16-pretrain-lvd1689m
```

模型 config 中：

```json
"model_type": "dinov3_vit"
```

### 8.3 输入

- RGBA 条件图像。
- 图像条件编码分辨率：512。
- sparse structure 分辨率：64。

主脚本参数：

```text
--trellis2-image-resolution 512
--trellis2-resolution 64
```

### 8.4 输出

输出 target sparse coords：

```text
coords: [N, 4]
```

坐标范围：

```text
batch: 0
x/y/z: 0..63
```

### 8.5 关键修复

之前错误地将 `--trellis2-resolution` 设置为 512。

TRELLIS2 的 `sample_sparse_structure(cond, resolution)` 中的 `resolution` 是 sparse grid 分辨率，不是图像分辨率。

错误会导致：

```text
RuntimeError: stride should not be zero
```

修复后：

```text
sparse_resolution = 64
image_resolution = 512
```

---

## 9. 阶段六：TRELLIS2 shape latent denormalization

### 9.1 为什么需要 denorm

Adapter 训练时的 TRELLIS2 target feats 是 normalized latent。

但是 TRELLIS2 decoder 期望输入的是 denormalized shape latent。

因此 adapter 输出在解码前必须执行：

```text
shape_slat = shape_slat * std + mean
```

### 9.2 使用 stats

stats 文件：

```text
dataset/partversexl/trellis2_shape_latents_stats.json
```

主 pipeline 默认启用：

```text
--denorm-shape-slat
```

对应参数：

```text
--trellis2-stats-json dataset/partversexl/trellis2_shape_latents_stats.json
```

### 9.3 不 denorm 的后果

如果不 denormalize，TRELLIS2 decoder 接收的 latent 分布错误，结果通常表现为：

- 几何破碎。
- 乱点云。
- 空洞严重。
- 结构不可识别。

---

## 10. 阶段七：TRELLIS2 shape latent flow refinement

### 10.1 使用模型

TRELLIS2 shape latent flow model：

```text
shape_slat_flow_model_512
```

当分辨率为 1024 时使用：

```text
shape_slat_flow_model_1024
```

### 10.2 原生 TRELLIS2 流程

TRELLIS2 原生 shape latent 采样流程是：

```text
random noise
  -> shape_slat_flow_model
  -> denormalized shape_slat
```

在 `Trellis2ImageTo3DPipeline.sample_shape_slat` 中，初始输入是随机噪声：

```text
noise = SparseTensor(feats=torch.randn(...), coords=coords)
```

### 10.3 当前实验性改法

当前实验路径将 adapter 输出作为 shape flow 的初始 latent：

```text
adapter predicted normalized shape_slat
  -> shape_slat_flow_model
  -> denormalized refined shape_slat
```

也就是说，adapter 不再直接输出最终 shape_slat，而是提供给 TRELLIS2 shape flow 的初始值。

### 10.4 参数

主 pipeline 参数：

```text
--refine-with-shape-flow
--shape-flow-steps 12
--shape-init-noise-strength 0.0
```

含义：

- `--refine-with-shape-flow`：启用 shape flow refinement。
- `--shape-flow-steps`：shape flow 采样步数。
- `--shape-init-noise-strength`：adapter 初值与随机噪声混合比例。

混合比例解释：

```text
0.0 = 完全使用 adapter 输出作为初值
0.3 = 70% adapter + 30% random noise
1.0 = 完全随机噪声，接近 TRELLIS2 原生 shape flow
```

### 10.5 当前结论

该方法已经可以端到端运行并导出 GLB。

但需要注意：

```text
TRELLIS2 官方采样器原本并未设计为从任意 adapter latent 起步。
```

因此该路径属于实验性 latent refinement。

如果使用随机初始化 old_slat 作为 adapter 输入，shape flow refinement 也无法弥补输入分布错误，最终质量仍然较低。

---

## 11. 阶段八：TRELLIS2 texture latent sampling

### 11.1 使用模型

TRELLIS2 texture flow model：

```text
tex_slat_flow_model_512
```

当分辨率为 1024 时使用：

```text
tex_slat_flow_model_1024
```

当前默认：

```text
512
```

### 11.2 输入

- 条件图像。
- TRELLIS2 image condition。
- shape_slat。

### 11.3 输出

输出 texture sparse latent：

```text
tex_slat
```

日志示例：

```text
Sampling texture SLat: 100%|...| 12/12
```

---

## 12. 阶段九：TRELLIS2 decode 与 GLB 导出

### 12.1 使用模型

TRELLIS2 decoder：

```text
pipeline.decode_latent(shape_slat, tex_slat, resolution)
```

### 12.2 输入

- denormalized shape_slat。
- sampled texture_slat。
- resolution 512。

### 12.3 输出

输出 mesh，并导出带纹理 GLB。

示例路径：

```text
outputs/fullpart_trellis2_toy_gun_trellis_sparse_coords/toy_gun/trellis2_textured.glb
```

训练集 seen sample 示例：

```text
outputs/adapter_seen_bridge/001e4438fc2742ad81356aae14c94d4a/pred_textured_000.glb
outputs/adapter_seen_bridge/001e4438fc2742ad81356aae14c94d4a/gt_textured_000.glb
```

### 12.4 GLB 导出工具

使用：

```text
o_voxel.postprocess.to_glb
```

主要步骤包括：

- mesh hole filling。
- remeshing。
- simplification。
- xatlas UV parameterization。
- texture baking。
- GLB export。

注意：

- 高复杂度 mesh 的 xatlas UV 展开非常慢。
- 批量生成 10 个高质量带纹理 GLB 可能耗时很长。

---

## 13. 主脚本

完整 pipeline 脚本：

```text
scripts/fullpart_to_trellis2_pipeline.py
```

主要功能：

- 读取输入图像与 box。
- 可选运行 FullPart Stage1。
- 可选运行 FullPart Stage2。
- 加载或复用已有 `global_slat.npz`。
- 可选随机或零初始化 old `global_slat`。
- 加载 adapter。
- 采样 TRELLIS2 target coords。
- adapter 转换 latent。
- 可选将 adapter 输出作为 `shape_slat_flow_model` 初值继续采样。
- 调用 TRELLIS2 texture/decode 导出 GLB。

### 13.1 使用已有 FullPart global_slat 的运行命令

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/miniconda3/envs/trellis2/bin/python scripts/fullpart_to_trellis2_pipeline.py \
  --image assets/demo_examples/toy_gun.png \
  --box assets/demo_examples/toy_gun.npy \
  --sample-id toy_gun \
  --output-dir outputs/fullpart_trellis2_toy_gun_trellis_sparse_coords \
  --use-existing-npz outputs/test_toy_gun/stage2/toy_gun/global_slat.npz \
  --adapter-ckpt /root/data/fullpart-main/exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
  --trellis2-root /root/data/TRELLIS.2 \
  --trellis2-pretrained /root/data/pretrained_models/trellis.2-4B/ckpts \
  --trellis2-config-file pipeline_local.json \
  --adapter-target-coords trellis2_sparse \
  --trellis2-resolution 64 \
  --trellis2-image-resolution 512 \
  --denorm-shape-slat \
  --trellis2-stats-json /root/data/fullpart-main/dataset/partversexl/trellis2_shape_latents_stats.json
```

### 13.2 随机初始化 old_slat + shape flow 的实验命令

该命令已经可以完整运行并导出 GLB，但质量较低，主要用于验证流程连通性：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native \
/opt/miniconda3/envs/trellis2/bin/python /root/data/fullpart-main/scripts/fullpart_to_trellis2_pipeline.py \
  --image /root/data/fullpart-main/assets/demo_examples/toy_gun.png \
  --box /root/data/fullpart-main/assets/demo_examples/toy_gun.npy \
  --sample-id toy_gun_init_old_slat_shape_flow \
  --output-dir /root/data/fullpart-main/outputs/init_old_slat_shape_flow_toy_gun \
  --init-global-slat \
  --init-global-slat-mode normal \
  --init-global-slat-num-tokens 20000 \
  --adapter-ckpt /root/data/fullpart-main/exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
  --adapter-target-coords trellis2_sparse \
  --refine-with-shape-flow \
  --shape-flow-steps 12 \
  --shape-init-noise-strength 0.0 \
  --trellis2-root /root/data/TRELLIS.2 \
  --trellis2-pretrained /root/data/pretrained_models/trellis.2-4B/ckpts \
  --trellis2-config-file pipeline_local.json \
  --trellis2-resolution 64 \
  --trellis2-image-resolution 512
```

成功输出示例：

```text
/root/data/fullpart-main/outputs/init_old_slat_shape_flow_toy_gun/toy_gun_init_old_slat_shape_flow/trellis2_shape_flow_textured.glb
```

---

## 14. 训练集 seen sample 桥接验证

### 14.1 验证目的

为了确认 adapter 桥接本身是否可行，使用 adapter 训练中见过的数据进行验证。

使用数据：

```text
old_slat: dataset/partversexl/textured_mesh_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/{asset_id}.npz
GT TRELLIS2 slat: dataset/partversexl/trellis2_global_shape_latents/{asset_id}.npz
condition image: dataset/partversexl/renders_cond/{asset_id}/000.png
```

### 14.2 验证样本

```text
001e4438fc2742ad81356aae14c94d4a
```

### 14.3 Adapter 数值指标

```json
{
  "asset_id": "001e4438fc2742ad81356aae14c94d4a",
  "old_tokens": 12463,
  "target_tokens": 12463,
  "mse_norm": 0.14174622297286987,
  "mae_norm": 0.26926302909851074,
  "cosine_norm": 0.9359874129295349,
  "pred_mean": 0.045680660754442215,
  "pred_std": 1.084984540939331,
  "gt_mean": 0.05069644749164581,
  "gt_std": 1.1721794605255127
}
```

解释：

- `cosine_norm = 0.936` 表明 adapter 预测 latent 与 GT latent 方向高度相似。
- `mse_norm = 0.142` 在 normalized latent 空间内较合理。
- `pred_std` 与 `gt_std` 接近，说明没有明显 latent collapse。

### 14.4 Seen sample 输出

Adapter 预测结果：

```text
outputs/adapter_seen_bridge/001e4438fc2742ad81356aae14c94d4a/pred_textured_000.glb
```

GT TRELLIS2 latent 对照结果：

```text
outputs/adapter_seen_bridge/001e4438fc2742ad81356aae14c94d4a/gt_textured_000.glb
```

### 14.5 使用训练数据 old_slat + shape flow 的推荐命令

当前推荐用 `batch_seen_bridge_trellis2.py` 验证质量，因为它读取 adapter 训练数据中的真实 old_slat：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native \
/opt/miniconda3/envs/trellis2/bin/python /root/data/fullpart-main/scripts/batch_seen_bridge_trellis2.py \
  --csv /root/data/fullpart-main/dataset/partversexl/home_sample_1000.csv \
  --old-dir /root/data/fullpart-main/dataset/partversexl/textured_mesh_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16 \
  --gt-dir /root/data/fullpart-main/dataset/partversexl/trellis2_global_shape_latents \
  --renders-cond-dir /root/data/fullpart-main/dataset/partversexl/renders_cond \
  --stats-json /root/data/fullpart-main/dataset/partversexl/trellis2_shape_latents_stats.json \
  --adapter-ckpt /root/data/fullpart-main/exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
  --output-dir /root/data/fullpart-main/outputs/adapter_seen_shape_flow_1 \
  --num-samples 1 \
  --start-index 0 \
  --view 000.png \
  --refine-with-shape-flow \
  --shape-flow-steps 12 \
  --shape-init-noise-strength 0.0 \
  --export-gt \
  --decimation-target 300000 \
  --texture-size 2048 \
  --trellis2-root /root/data/TRELLIS.2 \
  --trellis2-pretrained /root/data/pretrained_models/trellis.2-4B/ckpts \
  --config-file pipeline_local.json
```

输出包括：

```text
pred_textured_000.glb
gt_textured_000.glb
metrics.json
summary.json
```

### 14.6 对照实验建议

为了判断 `shape_slat_flow_model` refinement 是否有帮助，建议至少对比两组：

```text
A: old_slat -> adapter -> denorm -> texture/decode
B: old_slat -> adapter -> shape_slat_flow_model -> texture/decode
```

A 组命令去掉：

```text
--refine-with-shape-flow
```

B 组保留：

```text
--refine-with-shape-flow
```

如果 B 组质量低于 A 组，说明当前 TRELLIS2 shape flow 不适合直接把 adapter 输出作为初始状态继续积分，adapter 直接 decode 反而更稳。

### 14.7 结论

在训练分布内，即：

```text
训练集 old_slat + GT TRELLIS2 coords
```

adapter 可以预测接近 GT 的 TRELLIS2 shape latent，并能通过 TRELLIS2 decoder 导出带纹理模型。

这说明：

```text
FullPart latent -> TRELLIS2 latent 的桥接路径具备可行性。
```

---

## 15. 当前主要问题分析

虽然训练集 seen sample 验证通过，但对 toy_gun 这类完整生成结果质量仍可能较差。

当前主要原因不是 pipeline 不能运行，而是推理分布与 adapter 训练分布存在 gap。

### 15.1 Gap 1：old_slat 来源不同

Adapter 训练时输入：

```text
dataset precomputed old_slat
```

随机初始化实验路径输入：

```text
random initialized old_slat
```

旧版完整 pipeline 输入：

```text
FullPart Stage2 generated global_slat
```

二者分布可能不同。

### 15.2 Gap 2：target coords 来源不同

训练时：

```text
GT TRELLIS2 coords
```

实际推理时：

```text
TRELLIS2 sparse structure sampler coords
```

虽然后者是合法 TRELLIS2 coords，但不一定与训练 GT coords 完全一致。

### 15.3 Gap 3：shape flow 初值分布

TRELLIS2 原生 shape flow 默认从随机噪声开始。

当前实验路径从 adapter 输出开始：

```text
adapter predicted shape_slat -> shape_slat_flow_model
```

这并不是 TRELLIS2 官方默认采样路径，因此质量不一定稳定。

### 15.4 最可能影响质量的因素

排序如下：

1. 随机初始化 old_slat 与 adapter 训练 old_slat 存在严重 domain gap。
2. FullPart Stage2 generated global_slat 与 adapter 训练 old_slat 也可能存在 domain gap。
3. 推理时 sampled coords 与训练时 GT coords 不一致。
4. shape_slat_flow_model 从 adapter latent 起步属于实验性用法。
5. Adapter 使用 latent MSE 训练，未加入 decoder-aware 几何损失。
6. TRELLIS2 latent 空间对误差敏感，数值上接近不一定保证 mesh 完全稳定。

---

## 16. 已修复的问题汇总

### 16.1 DINOv3 加载问题

错误：

```text
Transformers does not recognize model_type 'dinov3_vit'
```

处理：

- 使用兼容 DINOv3 的 `transformers==4.57.3`。

### 16.2 diffusers 与 transformers 5 不兼容

错误：

```text
ImportError: cannot import name 'FLAX_WEIGHTS_NAME'
```

处理：

- 从 `transformers 5.x` 回退到 `4.57.3`。

### 16.3 FullPart UMT 旧代码导入问题

错误：

```text
ImportError: cannot import name 'apply_chunking_to_forward'
```

处理：

- 将 Stage1 / Stage2 config 改为 lazy import。
- adapter-only 或 `--use-existing-npz` 路径不再触发无关评估模块导入。

### 16.4 TRELLIS 子模块路径问题

错误：

```text
ModuleNotFoundError: No module named 'trellis'
```

处理：

```python
sys.path.insert(0, os.path.join(script_dir, "src/submodule/TRELLIS"))
```

### 16.5 RGB / RGBA 问题

错误：

```text
IndexError: index 3 is out of bounds for axis 2 with size 3
```

处理：

- 输入条件图统一转换为 RGBA。
- TRELLIS2 preprocess 中兼容 RGB rembg 输出。

### 16.6 batch size mismatch

错误：

```text
AssertionError: Batch size mismatch
```

处理：

```python
trellis2_slat.coords[:, 0] = 0
```

### 16.7 sparse resolution 错误

错误：

```text
RuntimeError: stride should not be zero
```

处理：

- sparse coords resolution 使用 64。
- condition image resolution 使用 512。

---

### 16.8 SparseTensor coords dtype 与 replace 接口问题

错误：

```text
AssertionError: only support int32
TypeError: SparseTensor.replace() missing 1 required positional argument: 'feats'
```

处理：

- 所有 sparse coords 显式转为 `torch.int32`。
- 对不支持只替换 `coords` 的 `SparseTensor.replace`，改为重新构造 `SparseTensor(feats=..., coords=...)`。

---

## 17. 推荐报告结论

可以在报告中总结为：

1. 本工作完成了 FullPart 到 TRELLIS2 的 latent bridge pipeline。
2. 通过 adapter 将 FullPart 8-channel global_slat 转换为 TRELLIS2 32-channel shape_slat。
3. 使用 TRELLIS2 sparse structure sampler 生成目标稀疏坐标，缓解坐标语义不一致问题。
4. 使用 TRELLIS2 shape latent normalization stats 对 adapter 输出进行 denormalization，使其符合 TRELLIS2 decoder 输入分布。
5. 在训练集 seen sample 上，adapter 预测 latent 与 GT TRELLIS2 latent 达到较高相似度，例如 cosine 约 0.936，说明桥接路径具备可行性。
6. 随机初始化 old_slat 路径已经可以端到端运行，但生成质量较低，原因是随机 latent 不在 adapter 训练输入分布内。
7. 使用 adapter 训练数据 old_slat 是当前更合理的质量验证路径。
8. 将 adapter 输出作为 `shape_slat_flow_model` 初值是一种实验性 TRELLIS2 latent refinement 方法，不属于 TRELLIS2 原生默认采样流程。
9. 后续优化方向包括：使用 Stage2 generated global_slat 参与 adapter 训练、加入 sampled coords 增强、加入 decoder-aware 几何损失，并系统评估 shape flow refinement 是否优于直接 decode。

---

## 18. 后续优化建议

### 18.1 缩小 old_slat domain gap

使用 FullPart Stage2 真实生成的 global_slat 作为 adapter 训练输入，而不是只使用 dataset precomputed old_slat。

### 18.2 target coords augmentation

训练时同时使用：

- GT TRELLIS2 coords。
- TRELLIS2 sparse sampler coords。
- remapped coords。

提升 adapter 对推理 coords 的鲁棒性。

### 18.3 decoder-aware loss

除了 latent MSE 外，引入：

- occupancy loss。
- mesh surface loss。
- normal consistency loss。
- decoder feature loss。

### 18.4 latent refinement

在 adapter 输出之后增加 TRELLIS2 latent refinement 或 flow denoising，使预测 latent 更接近 TRELLIS2 decoder manifold。

### 18.5 更高效批量可视化

高精度 textured GLB 导出非常慢，批量验证时建议先输出：

- normalized latent metrics。
- shape-only PLY。
- low-resolution / low-decimation GLB。

确认质量后再导出高质量 textured GLB。
