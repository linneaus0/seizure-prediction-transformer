# seizure-prediction-transformer

environment:  

Python==3.7.11  
torch==1.10.0+cu113  
pandas==1.3.4  
...  

把segmentation.csv seizure_summary.csv special_interictal.csv这三个文件存在CHBMIT数据集文件夹下

对病人数据集进行训练：  
首先下载CHB-MIT数据集，在SETTINGS_CHBMIT.json中更改数据集位置和预处理生成的cache保存位置。  
main.py中line68-80通过注释选择需要训练的病人(targets)，line123更改训练过程的存储位置  
在命令行输入：  
python main.py --mode cv --dataset CHBMIT  
对所选病人进行训练；cv: cross validation  

sample_CHBMIT.csv是所有病人上采样的比率  
struct.txt保存了所用网络的完整结构  
transformer.py是网络定义文件，包含vit网络  
Dataset.py是对CHBMIT数据集的实例化，但是后来并没有用到  
main.py包含了训练和测试代码以及后处理和结果打印，我设置的10个epoch存一个model  
myio文件可以不用看  
utils里存放的是数据集的预处理代码  
其中load_signals.py负责CHBMIT\FB\Kaggle三个数据集的预处理，只需看CHBMIT部分的  
load_signals_CHBMIT函数负责将选中的target的ictal(他这里的ictal其实就是我们的preictal，下文不做说明)和interictal的原始edf文件按指定通道(chs)读入，并在此时按照设置的SOP和SPH将符合条件的原始EEG拼接在一起，并存为np文件。  
Class PrepData 中的 preprocess函数将上一步处理得到的np文件进行切片，对于ictal数据按照sample_CHBMIT.csv进行overlapping，对于每个30s窗进行stft变换，然后把不要的频段滤除得到时频图，最后将x(data)和y(label)分别拼接起来。  
prep_data.py负责将单个病人的数据集变成n折并返回每一折的训练集、验证集、测试集。其中train_val_loo_split函数输入数据和标签，然后按发作次数n均分，取最后一次做测试，前面的数据按比例分成训练集和验证集，然后将ictal和interictal的数据concat在一起，去除测试集中overlap的数据，将数据按时间维度堆叠起来，最终输出[T,C,F]作为每一折的数据集。  

main.py运行中数据集处理的调用流程：  
main.py中line84: PrepData.apply()读入病人target和处理类型(ictal/interictal) -> load_signals_CHBMIT(line605)中read_raw_signals中的load_signals_CHBMIT(edf读入，预处理，拼接) -> load_signals_CHBMIT(line609)中preprocess(切片，overlapping) -> load_signals_CHBMIT(line610)save_hickle_file将时频图文件缓存 -> main.py(line96)train_val_loo_split将时频图处理成n折文件对应后续每一折的数据集  

整体运行流程：  
main.py -> 数据集预处理 -> 训练1折+验证 -> 下一折 ->...-> n折跑完+测试 -> 后处理+预测评估  
