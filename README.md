# WaveLip 训练模型代码

##Wav2Lip
Wav2Lip 是一种基于对抗生成网络的由语音驱动的人脸说话视频生成模型。如下图所示，Wav2Lip的网络模型总体上分成三块：生成器、判别器和一个预训练好的Lip-Sync Expert组成。网络的输入有2个：任意的一段视频和一段语音，输出为一段唇音同步的视频。生成器是基于encoder-decoder的网络结构，分别利用2个encoder: speech encoder, identity encoder去对输入的语音和视频人脸进行编码，并将二者的编码结果进行拼接，送入到 face decoder 中进行解码得到输出的视频帧。判别器Visual Quality Discriminator对生成结果的质量进行规范，提高生成视频的清晰度。为了更好的保证生成结果的唇音同步性，Wav2Lip引入了一个预预训练的唇音同步判别模型 Pre-trained Lip-sync Expert，作为衡量生成结果的唇音同步性的额外损失。

###1. 环境的配置
1.建议准备一台有显卡的linux系统电脑，或者可以选择使用第三方云服务器（Google Colab）

2.Python 3.6 或者更高版本
```code
ffmpeg: sudo apt-get install ffmpeg
```
3.必要的python包的安装，所需要的库名称都已经包含在requirements.txt文件中，可以使用
` pip install -r requirements.txt一次性安装.`

4.在本实验中利用到了人脸检测的相关技术，需要下载人脸检测预训练模型：`Face detection pre-trained model 并移动到 face_detection/detection/sfd/s3fd.pth`文件夹下.