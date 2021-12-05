# Models

## Decoder/Autoregressive Models
模型载入的代码好像都是:
- basic Model
- LMHeadModel
- ForSequenceClassification
    - Single-label
    - Multi-label

### OpenAI GPT
- **Position embeddings**: absolute position embeddings
- **Model objective**: Causal Language Modeling (CLM) objective
- **Good at**: predicting the next token in sequence


HG Models
- OpenAIGPTModel
    - bare transformer output raw hidden-states
- OpenAIGPTLMHeadModel
    - transformer with a language modeling head on top (linear layer with weights tied to the input embeddings)
- OpenAIGPTDoubleHeadsModel
    - transformer with a LM and Multiple-choice classification head on top
    - e.g. for RocStories/SWAG tasks (**SWAG**: Situations With Adversarial Generations)
        - commonsense inference
        - unifying natural language inference

### OpenAI GPT2
same as GPT

but,

there is a **Token Classification** for **NER**

### CTRL
Conditional Transformer Language 有条件的文本生成模型
- 写作模型
- 生成文本时可以指定文章类型，同一模型可以写作不同风格的文章
- CTRL关注了语料在不同场景中的不同含义
- CTRL可视为多任务学习

> CTRL核心思想：从无监督的海量数据中定位文章所在的领域，底层基于Transformer encoder。相对于之前的根据n-1个词计算第n个词的可能性，CTRL加入了条件c，即文章的类型作为控制信息。于是，在Attention计算过程中，类型与序列中的所有元素建立联系。**带条件的语言生成模型**

代码：
``https://github.com/salesforce/ctrl``

参考
[定向写作模型CTRL](https://zhuanlan.zhihu.com/p/100845592)

### Transformer-XL
要解决的问题：
> Transformer的编码能力超过了RNN，但是长距离依赖的建模能力不足。长文本序列被分为前后相接的片段，但是片段之间没有任何信息交互。如何赋予编码器捕获长距离依赖的能力。


提出：
- 片段级递归机制 Segment-level recurrence：利用缓存
- 相对位置编码机制 relative position embedding scheme，代替绝对位置编码

> Transformer-XL在训练的时候，也是以固定长度的片段输入，但是，它会将上一个片段的状态缓存，在计算下一段的时候，会重复使用上一个时间的隐层状态，所以Transformer-XL可以建模长期的依赖能力

- 每个时间片的预测都要从头开始，限制了应用场景
- 由于递归机制，推理段越长，提速越明显
- XLNet是以Transformer-XL为基础

### Reformer
> Reformer使用了许多方法来降低自回归transformer模型中内存和计算时间。
- 使用Axial position encoding，通过分解到更小的矩阵，来避免巨大的位置编码矩阵
- 使用LSH(local-sensitive hashing)注意力代替传统注意力，为避免计算整个的query-key乘法
- 使用可逆的transformer层来避免储存每个层的中间结果
- 以块的方式计算前馈操作，而不是整个batch
- 预训练**没有checkpoint**

可参考：
[对Reformer的深入解读](https://zhuanlan.zhihu.com/p/115741192)

### XLNet
提出问题：
> BERT很好，但不能用于生成。XLNET达到**双向学习**
- Permutation语言模型（PLM）
- 双流注意力机制
- 基于Transformer-XL，建立长依赖

PLM的贡献：
> 只要在Autoregressive中加入一个步骤，就能将AR和AE的优点统一起来。
- 随机取一句话的一种排列，将末尾的一些词遮盖掉
- 用AR的方式来预测被遮盖掉的词
- 有别于Bert的固定的MLM，具体实现方式为，对Attention Mask进行操作
    - 例如 1,2,3,4重新排列为3,2,4,1，这样最开始的1，通过预测，就能看到234。
    - **原句子不改变顺序**，因为permutation是通过mask来操作的

任务SOTA：
- SQuAD1.1&2.0: 问答
- RACE: 阅读理解
- MNLI：前提和假设关系的三分类：蕴含Entailment、矛盾Contradiction、中立Neutral
- QNLI：问题和句子蕴含/不蕴含二分类
- QQP：句子释义等效/不等效二分类
- RTE：句子1和2蕴含/非蕴含二分类
- SST-2：正面情感/负面情感而分类
- MRPC：是/非原句释义二分类
- CoLA：合乎语法/不合乎语法二分类
- STS-B：五分类任务/1-5相似性回归任务


## Encoder/Autoencoding Models

### BERT
Bidirectional Transformers for Language Understanding

- MLM - 15% tokens masked
    - special **mask token** with probability **0.8**
    - a **random token** different from the one masked with probability **0.1**
    - **same token** with probability **0.1**
- NSP
    - model must **predict the original sentence**
    - but a seconde objective: inputs are two sentences
        - 50% prob sentences are consecutive
        - model must **predict two sentences are consecutive or not**