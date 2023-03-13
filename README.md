# WaveLip 训练模型代码

##Wav2Lip
Wav2Lip 是一种基于对抗生成网络的由语音驱动的人脸说话视频生成模型。如下图所示，Wav2Lip的网络模型总体上分成三块：生成器、判别器和一个预训练好的Lip-Sync Expert组成。网络的输入有2个：任意的一段视频和一段语音，输出为一段唇音同步的视频。生成器是基于encoder-decoder的网络结构，分别利用2个encoder: speech encoder, identity encoder去对输入的语音和视频人脸进行编码，并将二者的编码结果进行拼接，送入到 face decoder 中进行解码得到输出的视频帧。判别器Visual Quality Discriminator对生成结果的质量进行规范，提高生成视频的清晰度。为了更好的保证生成结果的唇音同步性，Wav2Lip引入了一个预预训练的唇音同步判别模型 Pre-trained Lip-sync Expert，作为衡量生成结果的唇音同步性的额外损失。

本项目本质是在训练好的数据上做finetune，让你可以用自己的唇形数据
###1. 环境的配置
1.建议准备一台有显卡的linux系统电脑

2.Python 3.6 或者更高版本
```code
ffmpeg: sudo apt-get install ffmpeg
```
3.必要的python包的安装，所需要的库名称都已经包含在requirements.txt文件中，可以使用
` pip install -r requirements.txt一次性安装.`

4.在本实验中利用到了人脸检测的相关技术，需要下载人脸检测预训练模型：`Face detection pre-trained model 并移动到 face_detection/detection/sfd/s3fd.pth`文件夹下.

5.分别参考两个项目的代码来生成自己的代码：
一个是官方代码
```bash
git clone https://github.com/Rudrabha/Wav2Lip
```
一个是gitee上的项目，建议在开发时可以下载下来看看不同点
```bash
git clone https://gitee.com/sparkle__code__guy/wave2lip
```

###2. 训练前准备工作
1. 下载权重文件到`checkpoints`目录下,可以自行下载

|  模型  |	描述	 |  下载链接 |
|:-------:|:-------:|:-------:|
|  Wav2Lip  |Highly accurate lip-sync	|[wavlip](https://gitee.com/link?target=https%3A%2F%2Fiiitaphyd-my.sharepoint.com%2F%3Au%3A%2Fg%2Fpersonal%2Fradrabha_m_research_iiit_ac_in%2FEb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ%3Fe%3DTBFBVW)
|
|  Wav2Lip + GAN	|Slightly inferior lip-sync, but better visual quality	|[wavlip+GAN](https://gitee.com/link?target=https%3A%2F%2Fiiitaphyd-my.sharepoint.com%2F%3Au%3A%2Fg%2Fpersonal%2Fradrabha_m_research_iiit_ac_in%2FEdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA%3Fe%3Dn9ljGW)|
|  Expert Discriminator	|Weights of the expert discriminator	|[Expert Discriminator](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Flipsync%5Fexpert%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1)|

2. 下载人脸识别的[pre-trained model](https://gitee.com/link?target=https%3A%2F%2Fwww.adrianbulat.com%2Fdownloads%2Fpython-fan%2Fs3fd-619a316812.pth) 到`face_detection/detection/sfd/s3fd.pth`下。如果不能下载试试这个[链接](https://gitee.com/link?target=https%3A%2F%2Fiiitaphyd-my.sharepoint.com%2F%3Au%3A%2Fg%2Fpersonal%2Fprajwal_k_research_iiit_ac_in%2FEZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg%3Fe%3DqTasa8)

###3. 准备数据
准备自己的视频数据,至少要5个视频，视频中有明显的人的口型和声音。放入`data/original_data`目录下

###4.预处理数据
```bash
python preprocess.py --ngpu 1 --data_root E:/Projects/wav2lip_train/data/original_data --preprocessed_root E:/Projects/wav2lip_train/data/preprocessed_root --batch_size 8
```
data_root为原始视频地址，preprocessed_root为处理完的视频存放的位置
获取对应的文件列表并更新到filelists/train.txt和filelists/eval.txt。只保存对应的视频名称即可。
```code
from glob import glob
import shutil,os
result = list(glob("/home/guo/wave2lip/wave2lip_torch/Wav2Lip/data/preprocessed_root/original_data/*"))
print(result)
result_list = []
for i,dirpath in enumerate(result):
    shutil.move(dirpath,"./data/preprocessed_root/original_data/".format(i))
    result_list.append("{}".format(i))
print("\n".join(result_list))
```

###5.训练
执行下面的命令进行训练
```bash
python wav2lip_train.py --data_root ./data/preprocessed_root/data --checkpoint_dir ./savedmodel --syncnet_checkpoint_path ./checkpoints/lipsync_expert.pth
```

###6.模型预测
```bash
python inference.py --checkpoint_path ./savedmodel/checkpoint_step000000001.pth --face ./input/test.mp4 --audio ./input/audio.wav --out-file ./output
```
————————————————
####参考文档
CSDN博主「会发paper的学渣」的[原创文章](https://blog.csdn.net/sslfk/article/details/123419704) ,遵循CC 4.0 BY-SA版权协议


