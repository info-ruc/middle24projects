# 基于 Transformer 的中英文翻译模型

本项目使用 PyTorch 从零手写实现了一个基于 Transformer 的中英文翻译模型。通过训练该模型，您可以输入英文句子，模型将输出对应的中文翻译。本项目旨在深入理解 Transformer 模型的原理和实现细节。

## 目录

- [项目简介](#项目简介)
- [环境要求](#环境要求)
- [数据集](#数据集)
- [运行步骤](#运行步骤)
  - [1. 数据预处理](#1-数据预处理)
  - [2. 数据加载与分词](#2-数据加载与分词)
  - [3. 模型构建](#3-模型构建)
  - [4. 模型训练与验证](#4-模型训练与验证)
  - [5. 测试与推理](#5-测试与推理)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [注意事项](#注意事项)
- [改进建议](#改进建议)
- [参考资料](#参考资料)
- [许可证](#许可证)

## 项目简介

本项目手写实现了 Transformer 模型的各个组件，包括多头注意力机制、前馈神经网络、位置编码、编码器和解码器等。通过这个过程，我们深入理解了 Transformer 的内部机制和实现细节，并将其应用于中英文翻译任务。

## 环境要求

- Python 3.x
- PyTorch
- TorchText
- Pandas
- Scikit-learn

## 数据集

请将 `cmn.txt` 数据集放置在 `data/` 目录下。该文件包含了中英文句子对，每行一个句子对，英文和中文句子以制表符分隔。

您可以从以下链接下载数据集：[datasets](http://www.manythings.org/anki/)

## 运行步骤

### 1. 数据预处理

- **目的**：从原始数据集中提取英文和中文句子，并保存为单独的文件。
- **操作**：运行代码中的数据预处理部分，生成 `english_sentences.txt` 和 `chinese_sentences.txt`。

### 2. 数据加载与分词

- **目的**：加载预处理后的数据，对句子进行分词，并构建英文和中文的词汇表。
- **操作**：代码将自动执行分词和词汇表构建。

### 3. 模型构建

- **目的**：手写实现 Transformer 模型的各个组件，包括多头注意力、前馈神经网络、编码器、解码器等。
- **操作**：代码中已定义了模型的结构，包括各组件的实现。

### 4. 模型训练与验证

- **目的**：训练模型并使用验证集评估性能。
- **操作**：运行训练循环，默认训练 10 个 epoch。训练过程中将输出每个 epoch 的训练损失和验证损失。

### 5. 测试与推理

- **目的**：使用训练好的模型进行翻译测试。
- **操作**：在训练完成后，您可以输入英文句子，模型将输出对应的中文翻译。

## 使用方法

1. **克隆仓库**

   ```bash
   git clone https://github.com/xxuan66/Transformer--Translator.git
   cd Transformer--Translator
   ```

2. **安装依赖**

   请确保安装了以下依赖库：

   ```bash
   pip install torch torchtext pandas scikit-learn
   ```

3. **确保数据集存在**

   请确保 `data/cmn.txt` 文件存在。如未存在，请下载并放置在 `data/` 目录下。

4. **运行主程序**

   ```bash
   python main.py
   ```

5. **输入英文句子进行测试**

   在程序运行结束后，您可以按照提示输入英文句子，模型将返回中文翻译。

   ```
   请输入英文句子（输入 'quit' 退出）：How are you?
   中文翻译: 你好吗？
   ```

## 项目结构

```
├── data
│   ├── cmn.txt                # 中英文句子对数据集
│   ├── english_sentences.txt  # 提取的英文句子
│   └── chinese_sentences.txt  # 提取的中文句子
├── main.py                    # 主程序代码
└── README.md                  # 项目说明文件
```

## 注意事项

- **训练时间**：由于 Transformer 模型的复杂性，训练过程可能需要较长时间。建议在 GPU 环境下运行以加快训练速度。
- **模型效果**：由于模型的简单性和数据集的限制，翻译结果可能不够理想。您可以通过增加训练数据、调整模型参数等方法改进模型性能。
- **代码实现**：本项目手写实现了 Transformer 的各个组件，旨在帮助理解模型的内部机制。

## 改进建议

- **使用更大的数据集**：扩大训练数据量可以提升模型的泛化能力。
- **引入子词分词器**：使用 BPE（Byte Pair Encoding）等子词分词器可以减小词汇表大小，提高模型的泛化能力。
- **增加模型深度**：增加编码器和解码器的层数，提升模型的表达能力。
- **调整模型参数**：尝试不同的隐藏层维度、注意力头数、学习率等参数。
- **引入正则化技术**：如 Dropout、Early Stopping 等，防止模型过拟合。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [TorchText 文档](https://pytorch.org/text/stable/index.html)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

## 许可证

本项目仅供学习和研究使用。请勿将本项目用于任何商业用途。
