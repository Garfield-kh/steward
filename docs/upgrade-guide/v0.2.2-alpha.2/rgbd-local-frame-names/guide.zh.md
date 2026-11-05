---
kind: upgrade-guide
description: "Steward rgbd_video_gen 下载的本地文件名区分输入视角重建与目标视角预测。"
---

# RGBD 本地预测文件命名

[English](guide.md) | 中文

## 变更

Steward rgbd_video_gen 下载的 RGB 散图将含义不清的 `pred_<frame>.jpg` 改为 `source_reconstruction_<source>.jpg` 和 `target_prediction_<target>.jpg`。两组分别从零编号。8 个输入视角、1 个目标视角时，远端 `pred_08.jpg` 在本地保存为 `target_prediction_00.jpg`。

远端工件 `filename` 仍用于选择下载。结果工件、下载文件记录及本地清单文件提供 `local_filename`；`path` 和 `targets[].rgb_path` 指向更名后的本地文件。服务器原始 `result.json`、ZIP 内部文件名和 NPZ 深度键保留原标识及字节内容。

## 迁移

1. 将本地文件名拼接替换为工具返回的 `path` 或清单 `targets[].rgb_path`。调用 `rgbd_video_gen_download` 时继续传远端 `artifacts[].filename`。
2. 重新编译 Steward 插件。对已有成功任务再次下载，可在验证后将旧散图更名，无需再次传输。缺失或校验不通过的本地文件仍须正常下载并校验。仅查询结果不会更名历史文件。
3. 确认 8 个输入视角和 1 个目标视角对应本地 `source_reconstruction_00.jpg` 至 `source_reconstruction_07.jpg`，以及 `target_prediction_00.jpg`。目标深度键仍为 `v08`，应读取清单 `targets`，不能根据新的目标文件名推导深度键。
