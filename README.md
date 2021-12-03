# seizure-prediction-transformer

environment:
Python==3.7.11
torch==1.10.0+cu113
pandas==1.3.4
...

对病人数据集进行训练：
首先下载CHB-MIT数据集，在SETTINGS_CHBMIT.json中更改数据集位置和预处理生成的cache保存位置。
main.py中line68-80通过注释选择需要训练的病人(targets)
在命令行输入：
python main.py --mode cv --dataset CHBMIT
对所选病人进行训练；cv: cross validation
