# paddle
**paddlepaddle 相关项目**<br>

## 2020/08/31
**基于Paddlepaddle框架，以Cityscapes为数据集进行 DANet网络复现**<br>
cityscapesscripts文件夹为数据集处理代码，danet文件夹为基于paddlepaddle的复现代码，包括trainval、test、infer过程<br><br>

源项目链接：https://aistudio.baidu.com/aistudio/projectdetail/787016<br><br>

在使用paddle复现过程中，由于个人能力有限，复现j具体内容及改动部分如下：<br>
<br>
* 数据预处理<br>
	* 使用Cityscapes-gtFine数据集，以8：2比例划分train、val（test）数据集<br>
	* train过程768*768随机crop，val、test过程768*768中心crop<br>
	* 自定义trainid为20类<br>
* 网络构建：<br>
	* 使用resnet50作为backbone，各block输出通道数由[64, 128, 256, 512]改为[32, 64, 128, 256]<br>
	* CAM、PAM，input_chs由2048改为1024，inter_chs由input_chs//4改为512<br>
	* 输出结果包括CAM、PAM、CAM+PAM3部分<br>
* train、val过程<br>
	* batch size=8<br>
	* epoch = 30<br>
	* optimizer<br>
		* Momentum，momentum=0.9，l2_decay=0.0001<br>
	* learning rate<br>
		* 由1e-4线性warmup至1e-1，2epoch；<br>
		* 由1e-1按(1e-1-1e-4)*((1-iter/total_iter)**0.9)+1e-4的polynomial_decay方法进行变化，30epoch（其中iter表示单个batch过程，1epoch表示所有batch过程，polynomial_decay的前两个epoch的学习率由warmup替换）<br>
	* loss<br>
		* softmax_with_cross_entropy<br>
		* 对CAM、PAM、CAM+PAM以0.3、0.3、0.4的比例进行加权<br>
	* 保存val accuracy最高、val loss最低2个模型<br>
	* 输出train、val过程的accuracy、loss曲线<br>
* test过程<br>
	* 计算并输出混淆矩阵、各类iou、miou、fwiou等指标<br>
* infer过程<br>
	* 对图像进行推理预测


最终结果：best val acc 0.8627，miou 0.4063，fwiou 0.7784<br>
<br>
由于对网络进行了一些魔改，加上只进行了2次30epoch的训练，最终结果相较于论文有一定差距<br>
<br>
本复现结果只是对danet网络认识提供一种参考，希望大家可以一起交流学习<br>
