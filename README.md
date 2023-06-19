# NOLB-LongTail_Learning
旨在实现[论文](https://arxiv.org/abs/2303.03630)中的代码细节，以完成ML课设

## 环境
* python = 3.10(使用anaconda，请注意不要将其安装在名路径带空格的文件夹)

```
conda create -n ML python==3.10
conda activate ML
pip install -r requirements.txt
```
## 细节

* update(2023/6/16) 加入了论文作者的GML损失函数(很感谢)


$$\tilde{p}_j^i = \frac{N_j\mathrm{exp}(o_j^i)}{\displaystyle \sum_c^C N_c \mathrm{exp}(o_c^i)}$$

未完待续，先准备6级
