# `yellow_spoon` 全流水线命令

以下命令已根据 `.agent/` 中各阶段的最新记录核对。每个阶段使用独立的
uv 环境（`<package>/.venv/bin/python`），并从仓库根目录运行。

## 0. 视频预处理

```bash
process/.venv/bin/python -m process.cli \
  --video-path inputs/yellow_spoon.mp4
```

产物：`outputs/yellow_spoon/process/frames.json`、`video_info.json` 和
`frames/*.png`。

## 1. 手部重建（HaWoR）

```bash
hand_recon/.venv/bin/python -m hand_recon.cli \
  --frames-json outputs/yellow_spoon/process/frames.json
```

产物位于 `outputs/yellow_spoon/hand_recon/`，包括 `hands.json`、
`hand_anchors.json`、`meshes.npz` 和可视化视频。

## 2. 物体分割（SAM3）

```bash
segment/.venv/bin/python -m segment.cli --command video \
  --video.frames-json outputs/yellow_spoon/process/frames.json \
  --video.anchors-json outputs/yellow_spoon/hand_recon/hand_anchors.json \
  --video.sam-mask.checkpoint weights/sam3/sam3.pt \
  --video.sam-mask.text-prompts "yellow spoon" \
  --video.vis
```

必须显式指定仓库级 checkpoint；包内默认路径
`segment/ckpts/sam3/sam3.pt` 不存在。产物位于
`outputs/yellow_spoon/segment/`。

## 3. 单目度量几何（MoGe-3 ViT-giant）

```bash
geometry/.venv/bin/python -m geometry.cli --command video \
  --video.frames-json outputs/yellow_spoon/process/frames.json
```

产物位于 `outputs/yellow_spoon/geometry/`，包括 `geometry.json` 以及每帧的
深度、点图和内参。

## 4. 多视角物体重建（MV-SAM3D，推荐）

```bash
XFORMERS_DISABLED=1 obj_recon/.venv/bin/python -m obj_recon.cli \
  --masks-json outputs/yellow_spoon/segment/masks.json \
  --geometry-json outputs/yellow_spoon/geometry/geometry.json \
  --mv.enabled
```

`XFORMERS_DISABLED=1` 用于 RTX 50 系列（Blackwell）。默认均匀选取七个
关键帧；89 帧的 `yellow_spoon` 对应 `[0, 15, 29, 44, 59, 73, 88]`，从而提供
唯一的时间中间视角。每个物体输出一个
融合 mesh，并在 `layout.json` 中保存相对 view 0 的度量尺度：

```text
outputs/yellow_spoon/obj_recon/meshes/mv/yellow_spoon/
├── yellow_spoon.obj
├── yellow_spoon.glb
├── layout.json
└── view_poses.json
```

手动选择关键帧时，必须把所有值放在同一个 `--frame-index` 后面；tyro 对
重复的列表 flag 只保留最后一次：

```bash
XFORMERS_DISABLED=1 obj_recon/.venv/bin/python -m obj_recon.cli \
  --masks-json outputs/yellow_spoon/segment/masks.json \
  --geometry-json outputs/yellow_spoon/geometry/geometry.json \
  --mv.enabled --mv.keyframe-strategy manual \
  --frame-index 0 44 88
```

单视角备选方案（默认使用第 0 帧，输出到
`obj_recon/meshes/000000/yellow_spoon/`）：

```bash
XFORMERS_DISABLED=1 obj_recon/.venv/bin/python -m obj_recon.cli \
  --masks-json outputs/yellow_spoon/segment/masks.json \
  --geometry-json outputs/yellow_spoon/geometry/geometry.json
```

## 5. 6D 位姿估计（FoundationPose，多锚点）

```bash
pose_estimation/.venv/bin/python -m pose_estimation.cli \
  --frames-json outputs/yellow_spoon/process/frames.json \
  --masks-json outputs/yellow_spoon/segment/masks.json \
  --mesh-path outputs/yellow_spoon/obj_recon/meshes/mv/yellow_spoon/yellow_spoon.obj \
  --temporal-filter.enabled \
  --vis
```

对于 MV mesh，程序从 mesh 同目录的 `view_poses.json` 读取所有重建视角，选择
时间顺序上的中间视角，并使用该视角的重建输出位姿初始化 FoundationPose，再分别
向视频两端逐帧跟踪；不会在视角之间做 SE(3) 插值。prompt 默认由 mesh 文件名推导
（`yellow_spoon` → `yellow spoon`），度量尺度默认读取同目录的 `layout.json`。
可分别用以下参数覆盖：

```text
--prompt-id "yellow spoon"
--mesh-scale <meters-per-unit>
```

`--anchor-frames` 仅用于非 MV mesh；MV 模式始终使用上述中间重建视角。默认
逐帧 refinement 为 10 次、crop ratio 为 2.0；此外用 object mask 内的 MoGe
点云中心约束平移，避免快速运动时离开 FoundationPose crop。旋转仍由
FoundationPose 估计。`--temporal-filter.enabled` 会在跟踪后运行常速度误差状态
EKF，并用 mask/depth 质量与创新门控拒绝勺子对称性造成的旋转翻转；原始位姿保留
在 `poses/`，滤波结果写入 `poses_filtered/`，后续对齐会优先选择滤波结果。

单视角模式应把 `--mesh-path` 指向
`obj_recon/meshes/000000/yellow_spoon/yellow_spoon.obj`；此时从
`--init-frame`（默认 0）开始双向跟踪。

`--vis` 生成单个 `outputs/yellow_spoon/pose_estimation/vis.mp4`，编码为
H.264/yuv420p。FoundationPose 不会跨视角拟合尺度，因此必须使用由
`layout.json` 标定的度量 mesh。

## 6. 可选手物校正

```bash
hand_object_alignment/.venv/bin/python -m hand_object_alignment.cli \
  --clip-root outputs/yellow_spoon \
  --mode auto_per_frame
```

三种模式：

* `--mode auto_per_frame`（默认）：对每一帧独立拟合一个刚性校正。先用
  平移网格搜索做 warmup，再做 6-DoF Powell 精化，目标为 Huber 化的
  手–物接触距离 + 穿透惩罚 + 信任域先验。
* `--mode auto_global`：所有 in-hand 帧共享同一个校正（do-as-i-do 的
  全轨迹 warmup 的对应物）。
* `--mode manual`：自由给定的全局 6-DoF 校正，与旧版兼容——平移为米，
  旋转为弧度 axis-angle。

自动模式只有当每一道验收门通过才会写 `status: accepted, usable: true`
的 manifest：校正后中位接触距离 ≤ `--contact-dist-m`（默认 2 cm）、
穿透深度 ≤ `--max-penetration-m`（默认 5 mm）、中位距离不劣化、
|平移| ≤ `--max-translation-m`、|旋转| ≤ `--max-rotation-deg`、且至少有
`--min-inhand-overlap-frames` 帧同时具备位姿和有效手部。任何一道门
失败都会写 rejected manifest 并清空 `poses/`，下游永远不会消费一个
坏的自动校正。

## 7. MuJoCo 场景构建

```bash
scene_construction/.venv/bin/python -m scene_construction.cli \
  --clip-root outputs/yellow_spoon \
  --object-trajectory auto
```

`--object-trajectory` 精确取值为 `auto|canonical|aligned`：`auto`（默认）仅在
可选 manifest 明确 accepted 且通过校验时采用校正轨迹，否则在 manifest
缺失、disabled 或 rejected 时沿用旧的 `pose_estimation` 轨迹；`canonical`
始终选原始轨迹；`aligned` 强制要求可用校正轨迹。还可用
`--alignment-manifest /absolute/or/relative/poses.json` 自由选择 manifest。

该阶段读取上述各阶段产物，自动判断 `left`、`right` 或 `bimanual` 手型，
并把结果写入 `outputs/yellow_spoon/scene_construction/`。后续主要输入位于：

```text
outputs/yellow_spoon/scene_construction/sharpa/<hand-type>/yellow_spoon/0/
```

## 8. 机器人重定向

```bash
retarget/.venv/bin/python -m retarget.cli \
  --task yellow_spoon \
  --save-video
```

对于本视频，默认 `output_root` 已指向
`outputs/yellow_spoon/scene_construction`。主要产物为
`trajectory_kinematic.npz`、`scene.xml` 和 `scene_eq.xml`。

## 9. 物理优化（MuJoCo Warp MPC）

```bash
physics_opt/.venv/bin/python -m physics_opt.cli \
  --task yellow_spoon \
  --no-show-viewer \
  --no-wait-on-finish \
  --save-video
```

对于自动识别为双手的 `yellow_spoon`，最终产物位于：

```text
outputs/yellow_spoon/scene_construction/sharpa/bimanual/yellow_spoon/0/physics_opt/
├── trajectory_mjwp.npz
└── config.yaml
```

## 路径注意事项

- 各阶段统一写入 `outputs/yellow_spoon/<stage>/`。
- `scene_construction` 的 `<hand-type>` 由有效手部 mask 自动决定，不应在尚未
  运行时假定一定是 `bimanual`。
- `retarget` 与 `physics_opt` 当前默认 `output_root` 硬编码为
  `outputs/yellow_spoon/scene_construction`，所以本示例无需额外参数。处理其他
  clip 时必须同时显式传入 `--task <clip>` 和
  `--output-root outputs/<clip>/scene_construction`，并保证手型和机器人类型与场景
  构建阶段一致。
- 从其他工作目录运行时，请把输入和输出路径改为绝对路径。
