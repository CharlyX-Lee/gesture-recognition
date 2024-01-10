# 手势识别检测——基于YOLO_V5算法模型剪枝及推理



## 1 环境准备



### 1.1 工具及环境要求

#### 工具

- [labelimg](https://pypi.org/project/labelImg/)

- [AI Studio](https://aistudio.baidu.com/aistudio/index)
- [YOLO2COCO](https://gitee.com/RapidAI/YOLO2COCO.git)
- [PaddleUtils](https://gitee.com/stark-lin/paddleutils.git)
- [paddleyolo](https://gitee.com/lrp114/PaddleYOLO.git)

#### 本地环境要求

- openvino==2022.2.0
- paddle2onnx==1.0.5
- paddlepaddle==2.4.2
- opencv-python==4.2.0.32
- onnx==1.11.0
- tensorflow==2.9.1



### 1.2 工具介绍

#### 1.2.1 [python](https://www.python.org/)

###### 可以安装python到本地，也可以直接跳过本步，去anaconda创建python的虚拟环境

#### 1.2.2 [anaconda](https://www.anaconda.com/)

anaconda仅作为虚拟环境，会创建及进入虚拟环境即可，此处不需要费时间

###### HINT:建议虚拟环境安装python3.7.9版本，此版本经本人实测无问题，其余版本不敢保证后续依赖项是否出错。

自学参考教程（仅供参考，本人非本人严格审核过文章）：

[Anaconda介绍、安装及使用保姆级教程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/476403161?utm_id=0)

[安装conda搭建python环境（保姆级教程）_conda创建python虚拟环境-CSDN博客](https://blog.csdn.net/Q_fairy/article/details/129158178)

[Anaconda安装（Python） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/347990651)

#### 1.2.3 [labelimg](https://so.csdn.net/so/search?q=labelimg&spm=1001.2101.3001.7020)

  labelimg是一个有图形界面的图像标注工具，用来给数据打标签。

```anaconda环境
conda activate [虚拟环境名称] #激活anaconda虚拟环境
pip3 install labelimg #安装labelimg
pip3 install labelimg -i https://pypi.tuna.tsinghua.edu.cn/simple
#上一个命令如果速度慢，可以试着开VPN，速度应该会快，无VPN可使用国内清华镜像，即https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 1.2.4 [AI Studio](https://aistudio.baidu.com/index)

###### 打个广告，帮我助力一下，我的算力用完了

###### AI Studio学习与实训社区上线 Tesla A100！为我助力赢10点免费算力，助力成功你可领100点算力卡哦～https://aistudio.baidu.com/aistudio/newbie?invitation=1&sharedUserId=3735541&sharedUserName=CharlyX%20Lee

AI Studio是基于百度深度学习平台飞桨的人工智能学习与实训社区，提供在线编程环境、免费GPU 算力、海量开源算法和开放数据，帮助开发者快速创建和部署模。初次使用的小伙伴记得注册之后完成新手礼包获取算力卡。

![在这里插入图片描述](https://img-blog.csdnimg.cn/9e001b60c1e34e30b5398a7e1c661037.png)

完成任务过后，你也可以拥有和很多的GPU使用时间。

![image-20240110162228571](C:\Users\12637\AppData\Roaming\Typora\typora-user-images\image-20240110162228571.png)



#### 1.2.5 YOLO2COCO

因为我们拿到的数据集是yolo格式的，所以需要借助这个工具把YOLO格式的标签数据转成COCO格式数据集。

#### 1.2.6 PaddleUtils

这是一个paddlepaddle模型剪枝工具，我们需要把训练得到的ppyoloe手势检测模型进行裁剪。

#### 1.2.7 paddleyolo

paddleyolo里面有很多目标检测算法，其中包括ppyoloe这个算法，使用的时候只需要配置一些文件就可以训练我们的模型了，非常方便。因为训练模型需要GPU，所以需要在AI Studio里面使用paddleyolo。



### 1.3 库的安装

```anaconda环境配置
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openvino-dev[ONNX,pytorch,tensorflow]==2022.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

###### 本人记不清所有的环境配置有哪些，后续运行过程中，如果发现有依赖包缺失及时安装即可。



## 2 数据准备

### 2.1 数据介绍

  目录结构
----dataset
      ----000-one
            ----gesture-one-2021-03-07_23-07-48-2_61613.jpg
            ----gesture-one-2021-03-07_23-07-48-4_40096.jpg
       ·      
       ·     
       ·     
      ----013-pink
            ----gesture-pink-2021-03-07_23-07-55-1_39459.jpg
            ----gesture-pink-2021-03-07_23-07-55-2_5978.jpg
            .
            .
            .

  图片数量有2017张，每类手势的数据集数量不一。图片大致如下所示：


![image-20240110165808561](C:\Users\12637\AppData\Roaming\Typora\typora-user-images\image-20240110165808561.png)

数据来自校赛官方提供的数据集，我们将使用labelimg对数据进行标注，即打标签。

### 2.2 数据标注

```
conda activate [环境名称] #打开anaconda虚拟环境

labelimg #启动labelimg
```

![image-20240110170106674](C:\Users\12637\AppData\Roaming\Typora\typora-user-images\image-20240110170106674.png)

指定要标注的图片文件夹，我们的文件夹是dataset。指定新建一个labels文件夹，被标注的图片生成的.txt标签文件会自动保存到labels文件夹下。

![image-20240110170338166](C:\Users\12637\AppData\Roaming\Typora\typora-user-images\image-20240110170338166.png)

![image-20240110170428891](C:\Users\12637\AppData\Roaming\Typora\typora-user-images\image-20240110170428891.png)

数据格式选用yolo，因为公开数据集是yolo格式，所以统一用yolo。图片中是手势“1”，所以数据标注为000-one。

![image-20240110170525717](C:\Users\12637\AppData\Roaming\Typora\typora-user-images\image-20240110170525717.png)

生成的标签txt文件。除了根据图片名字生成相应的标签txt，还有一个classes.txt。

![image-20240110170635588](C:\Users\12637\AppData\Roaming\Typora\typora-user-images\image-20240110170635588.png)

本次手势识别共有14中标签，classes.txt如下所示：

![image-20240110170704039](C:\Users\12637\AppData\Roaming\Typora\typora-user-images\image-20240110170704039.png)

生成的标签txt文件如下图所示：

![image-20240110170844082](C:\Users\12637\AppData\Roaming\Typora\typora-user-images\image-20240110170844082.png)

0,1,...,13表示标签号，图片中标注了多少个框，相应的生成几排数据。如下图中，假如图片是680x640，0或者1后面的几个小数分别代表框的中心坐标x，y，框的宽w，框的高h。为什么是小数？是因为x/680，y/640，w/680，h/640。

![image-20240110171013584](C:\Users\12637\AppData\Roaming\Typora\typora-user-images\image-20240110171013584.png)



我们先做到这一步，后续本人有时间继续更新~

### 2.3 数据转换

数据集拆分，严格意义上讲，应该把数据拆成训练集、测试集、验证集，我们为了方便，就只需要拆分成训练集和验证集就可以了。在YOLO2COCO\dataset文件下建立一个yolo_mask文件夹，然后将dataset文件夹和label文件夹移动到此文件夹下。点击[YOLO2COCO](https://gitee.com/RapidAI/YOLO2COCO.git)获取工具。

