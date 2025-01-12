# 数字信号处理期末大作业-第6组

# FFT等经典算法复现

见`ReproducedAlgorithm`文件夹

## 使用传统打分模型的检索任务

见Retrieve

sound_retrieval.py文件用传统的FFT和STFT以及MFCC方法提取特征，然后通过cosine similarity计算相似度来检索声音。

classification_retrieval.py文件用BEATs和CLAP等分类模型，经过修改后用于检索任务。

evaluate_results.py文件包含实验使用到的评价指标。

可以通过run.sh脚本一键测试，命令如下。

`bash run.sh`

## 使用机器学习来做分类检索

传统的CNN等网络代码，训练代码以及测试代码已经放至`TraditionalClassification`文件夹下，包含分类和检索两个任务。

我们还使用了预训练的BEATs， CLAP。

见以各模型名称命名的文件夹
