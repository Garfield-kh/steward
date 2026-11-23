# 装配参考夹具

[English](README.md) | 中文

`upstream-defaults.json` 冻结机器人与 Cobot 模型、运行时与紧固默认值、带抽屉的 sideboard、螺钉驱动调用轨迹、31 个默认分支 AST 指纹及 Harness 工具/相机参考。它由公开上游 `b0af02e596da18316ca815f15b89a23ea2b55608` 生成，并非由装配实现生成。唯一源码修正是已说明的无抽屉 sideboard 返回语句缩进。`source` 元数据仅作描述，不构成 Git 依赖。

有意重新生成时，从仓库根目录执行以下命令（需要 Python 3.11、NumPy、pytest 和 Node.js）：

```sh
mkdir -p steward/var/upstream-reference
git archive b0af02e596da18316ca815f15b89a23ea2b55608 steward/src \
  steward/dsh/packages/dsh-tool-ego/src steward/dsh/packages/dsh-ego-runtime-client/src \
  | tar -x -C steward/var/upstream-reference
PYTHONPATH=steward/src python steward/tests/fixtures/generate_upstream_defaults.py \
  steward/var/upstream-reference
```

只有上述手动生成步骤需要上游历史，测试不会调用它。重新生成的期望值必须独立审查，不能靠更新参考让实现回归通过。生成器从导出的上游树读取参考实现，执行其构造函数和驱动例程，并使用与测试相同的观察过程及 AST 投影。AST 指纹使用 Python 的 AST 表示，并忽略 Python 3.12 引入的空类型参数列表。本参考由 Python 3.11 生成和验证。
