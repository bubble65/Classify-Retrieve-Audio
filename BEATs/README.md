# BEATs使用

## BEATs的预训练过程

![image-20250112160816320](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250112160816320.png) 

## 我们把微调后的BEATs用于分类/检索任务

![image-20250112160928734](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250112160928734.png) 

![image-20250112160948195](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250112160948195.png) 

use_beats.py是我们开发的，其余来自BEATs的开源仓库[unilm/beats at master · microsoft/unilm · GitHub](https://github.com/microsoft/unilm/tree/master/beats)

use_beats.py可以用来微调(train_mode = True)

```python
@dataclass
class TrainArguments:
    """
    Arguments related to training.
    """
    data_dir: str = field(default='../ESC-50-master/audio')
    csv_file: str = field(default='../ESC-50-master/meta/esc50.csv')
    batch_size: int = field(default=16)
    learning_rate: float = field(default=1e-4)
    num_epochs: int = field(default=20)
    pretrained_path: str = field(default="../ckpt/BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt2.pt")
    output_model_path: str = field(default="../ckpt/finetuned_model_1.pt")
    train_mode: bool = field(default=False)

@dataclass
class ModelArguments:
    """
    Arguments related to model initialization and configuration.
    """
    checkpoint_path: str = field(default='../ckpt/finetuned_model_1.pt')
```

也可以用来测试模型（train_mode = False）

其中检索任务会生成一个jsonl文件, class是query属于的声音种类（ESC-50标准下），列表是相似度从高到底排序后的前20个声音的种类。

```python
{"query": [3, 11, 2, 2, 2, 2, 46, 9, 46, 26, 29, 2, 6, 6, 3, 29, 23, 2, 3, 5], "class": 2}
```

