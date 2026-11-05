---
kind: upgrade-guide
description: "Steward 模型下载使用工作空间输出目录，并将提交输入与请求恢复状态分别保存。"
---

# Steward 模型输入与输出目录

[English](guide.md) | 中文

## 变更

可选的 Steward rgbd_video_gen 和 NovelAnything 客户端将模型输入保存在机器人工作空间的 `inputs/<model>/` 下，结果保存在 `outputs/<model>/<job_id>/` 下。工具返回 `local_manifest_path`，指向包含下载路径和提交输入引用的本地清单。请求恢复状态与模型产物分别保存。这些集成不安装 SLARM。

提交会将 JSON 场景与参数或上传 ZIP 归档到对应的 `inputs/<model>/` 目录。手爪视角准备默认写入 `inputs/rgbd_video_gen/prepared/`。服务器 demo 保留选择信息，不包含服务器持有的证据图像。缺少本地输入记录的历史提交，其输入引用为 null。已有旧文件不会被删除或自动移动。

## 迁移

1. 在机器人 Ego profile 中安装需要的 bundle，编译 Steward 插件并重启机器人 Harness。通过模型客户端的 `inputsDir` 和 `outputsDir` 覆盖默认目录；相对路径从 `inputRoot` 解析。任务 ID 可能冲突的不同端点需要独立输出目录。
2. 后续流程改用工具返回的 `local_manifest_path`，不再拼接路径。读取 `files` 获取已校验下载；读取 RGBD `targets` 获取目标相机、RGB 与深度数组的对应关系。路径为 null 表示尚未下载该文件。
3. 查询并下载已有成功任务，将其发布到新目录；或复制历史文件，并在使用前按服务器结果进行校验。在后续流程使用新清单之前，保留已有文件。新提交会在 POST 前保留输入。
4. 确认下载任务在 `outputs/<model>/` 下包含 `result.json`、`manifest.json` 和请求的文件。现有 `var/` Git 忽略规则覆盖调用方工作空间；运行数据和凭据不得提交。
