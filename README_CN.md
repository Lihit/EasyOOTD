## EasyOOTD: 姿态可控的虚拟穿衣
### EasyOOTD的特色
* 基于SD1.5模型新增了一个Adapter实现虚拟穿衣, 大小只有198 MB，可适配不同的基座模型。
* 无缝接入预训练好的ControlNet,可以实现姿态可控的虚拟穿衣。
* 针对该Adapter训练了一个LCM-Lora，大小只有76 MB，可在4步完成推理。
* 设计了一个基于SAM的可交互式抠图的webui，方便用户自己指定虚拟穿衣的替换区域。

**注意：该项目是出于兴趣爱好在业余时间完成，因为只增加了一个Adapter，所以模型的能力有其局限性，如果你将它直接应用到产品，你将自己承担风险**

### 模型结构
<img src="assets/introductions/model.jpg" alt="视频号" width="768" height="490">

### 如何使用
* 环境安装
* 模型下载
* 运行和使用教程

### 关于我
欢迎关注我的视频号，会持续分享我做的AIGC的内容。有合作需求欢迎私信。

<img src="assets/introductions/shipinhao.jpg" alt="视频号" width="300" height="350">