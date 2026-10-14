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


## 推理

在 Project 目录运行：

```powershell
python -B demo.py --port 7863
```

网页中上传原图，绘制编辑区域或上传黑色编辑、白色保留的区域图，再输入指令。也可直接运行：

```powershell
python -B infer.py --source "data/001000/001612/source.png" --mask "data/001000/001612/mask.png" --prompt "Put an oven next to the microwave."
```

输出到 output 下的新目录：source.png、mask.png、prompt.txt、result.png。种子、步数和 CFG 参数嵌入结果 PNG。

## 训练

当前默认划分已匹配本端实际数据：

```powershell
python -B OminiSketch/train/train_lora.py --name new_trial --samples 128
```

省略 --samples 使用训练分区全量样本。默认读取已更新的 train/split_manifest.json；若显式传 --split-manifest，指定的文件仍须与本端数据完全匹配。

默认 rank16、lr=1e-4、4轮、累积1、gate0.1、文字 LoRA 开启、感知权重0.5/grid4。需要暖启动时显式传 --initial-lora，否则新初始化 LoRA。新产物写入 model/skelora/<name>/，新配置写入 train/configs/，不覆盖同名实验。

训练后将 config.yaml 的 models.sketch_lora 指向选中的 adapter 目录，再重新启动推理进程。

## 依赖与数据流

- [推理端说明及完整图](OminiSketch/core/readme.md)
- [训练端说明及完整图](OminiSketch/train/readme.md)

