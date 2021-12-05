# Introduction

## GLUE
General Language Understanding Evaluation
> 目的：NLU九项任务基准数据集

参考：
[GLUE基准数据集介绍及下载](https://zhuanlan.zhihu.com/p/135283598)
[GLUE官网](gluebenchmark.com/)

|数据集|分类|任务|领域|包含|
|----|----|----|----|----|
|CoLA（The Corpus of Linguistic Acceptability 语言可接受性语料库）|单句分类|可接受性、合乎语法/不合乎语法二分类|misc.|训练集8551, 开发集1043，测试集1063|
|SST-2（The Stanford Sentiment Treebank 斯坦福情感树库）|单句分类|正面情感/负面情感而分类|电影评论|训练集67350, 开发集873，测试集1821|
|MRPC（The Microsoft Research Paraphrase Corpus，微软研究院释义语料库）|相似性和释义|是/非原句释义二分类|新闻|训练集3668, 开发集408，测试集1725|
|STS-B（The Semantic Textual Similarity Benchmark，语义文本相似性基准测试）|相似性和释义|五分类任务/1-5相似性回归任务|misc.|训练集5749, 开发集1379，测试集1377|
|QQP（The Quora Question Pairs, Quora问题对数据集）|相似性和释义|句子释义等效/不等效二分类|社交QA|训练集363870, 开发集40431，测试集390965|
|MNLI（The Multi-Genre Natural Language Inference Corpus, 多类型自然语言推理数据库）|推理Inference|NLI，前提和假设关系的三分类：蕴含Entailment、矛盾Contradiction、中立Neutral|misc|训练集392702, 开发集匹配9815/非匹配9832，测试集匹配9796/非匹配9847|
|QNLI（Qusetion-answering NLI，问答自然语言推断）|推理Inference|问题和句子蕴含/不蕴含二分类|Wiki|训练集104743, 开发集5463，测试集5461|
|RTE（The Recognizing Textual Entailment datasets，识别文本蕴含数据集）|推理Inference|句子1和2蕴含/非蕴含二分类|新闻、Wiki|训练集2491, 开发集277，测试集3000|
|WNLI（Winograd NLI，Winograd自然语言推断）|推理Inference|共指/句子1和2蕴含/非蕴含二分类|小说书籍|训练集635, 开发集71，测试集146|