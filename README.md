# OminiSketch

通过原图、编辑区域和文本指令，使用 FLUX Dev + skelora 在指定区域生成内容，并在采样过程中保护背景。

## 目录

```text
Project/
  demo.py                       网页推理入口
  infer.py                      命令行推理入口
  requirements.txt              运行依赖
  README.md
  OminiSketch/
    config.yaml                 数据、模型和输出路径
    core/                       四个推理模块、包入口、说明与依赖/数据流图
    train/                      train_lora.py、helpers.py、划分、说明与依赖/数据流图
  data/                         原始训练样本
  model/
    flux_dev/                   FLUX Dev 基座
    Redux/image_encoder/        训练用冻结 SigLIP 教师
    skelora/                    推理 adapter 与新训练输出
  output/                       推理结果
  Test/                         划分/校验脚本、备份和报告（非运行依赖）
```

搬迁报告、检查脚本和图生成器不属于运行依赖，已移出 Project 到 `D:/plans/projects/result/cleanup_20261014/`。删除 result 不影响本项目代码。

## 运行前提

1. Python 3.12 或更新版本；当前代码使用了新式 f-string 语法。
2. 安装与显卡驱动匹配的 CUDA PyTorch，以及 requirements.txt 中依赖。依赖安装和真实 GPU 运行尚未在本机验证。
3. 将兼容的 FLUX Dev 完整组件放到 config.yaml 的 `models.flux_base`，不要用 Schnell 替代。当前本地该目录为空。
4. 推理 adapter 目录必须有 adapter_model.safetensors、adapter_config.json、training_protocol.json；当前 skelora 三个文件已保留。
5. 默认训练感知权重为 0.5，需要 config.yaml 的 `models.redux_image_encoder` 指向完整 SigLIP 教师；当前本地目录为空。设 `--perceptual-weight 0` 可不加载教师，但会改变训练损失，不能视为默认方案的等价复现。
6. split_manifest.json 已按本端实际存在的 4001 例重新分组划分，训练 2801、验证 800、测试 400。同一归一化原图分组不跨分区，旧名单的缺失项已去除；原始数据未删除。

因此 GPU 不是唯一前提；完整模型、运行依赖和匹配的训练划分也必须准备好。

## 推理

在 Project 目录运行：

```powershell
python -B demo.py --port 7863
```

网页中上传原图，绘制编辑区域或上传黑色编辑、白色保留的区域图，再输入指令。也可直接运行：

```powershell
python -B infer.py --source "data/001000/001612/source.png" --mask "data/001000/001612/mask.png" --prompt "Put an oven next to the microwave."
```

输出到 output 下的新目录：source.png、mask.png、prompt.txt、result.png。种子、步数和 CFG 参数嵌入结果 PNG。推理不使用 Redux，也不在结果端拼接或缩放贴图。

## 训练

当前默认划分已匹配本端实际数据：

```powershell
python -B OminiSketch/train/train_lora.py --name new_trial --samples 128
```

省略 --samples 使用训练分区全量样本。默认读取已更新的 train/split_manifest.json；若显式传 --split-manifest，指定的文件仍须与本端数据完全匹配。

默认 rank16、lr=1e-4、4轮、累积1、gate0.1、文字 LoRA 开启、感知权重0.5/grid4。需要暖启动时显式传 --initial-lora，否则新初始化 LoRA。新产物写入 model/skelora/<name>/，新配置写入 train/configs/，不覆盖同名实验。

训练后将 config.yaml 的 models.sketch_lora 指向选中的 adapter 目录，再重新启动推理进程。训练集用于更新参数，验证子集用于选优，测试集不参与训练和选优。

## 依赖与数据流

- [推理端说明及完整图](OminiSketch/core/readme.md)
- [训练端说明及完整图](OminiSketch/train/readme.md)

