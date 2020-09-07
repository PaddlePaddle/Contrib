
cityscape数据集官方处理代码微调版本，适用于gt-fine数据集，具体调整及说明如下


gt-fine数据集，gt中1个样本gt包括4个文件
* color:  彩色类别图（用于可视化观察）
* instanceIds:  实例分割ID
* labelIds:  标签ID（35小类）
* polygons:  边界多边形（35小类及对应边界像素位置）

其中的**labelIds**可以用作**图像分割**，但是35类可能不符合个人训练要求，可以对类别进行精简

具体操作：
1. *打开/home/aistudio/work/paddle/cityscapesscripts/preparation/createTrainIdLabelImgs.py文件，修改：*
	*  ***25行 datapath = r'/home/aistudio/work/paddle/cityscapesscripts/'***
	*  ***27行 sys.path.append( os.path.normpath( os.path.join( os.path.dirname( datapath ) , 'helpers' ) ) )，并添加一行sys.path.append( os.path.normpath( os.path.join( os.path.dirname( datapath ) ) ) )***
	*  ***40行 cityscapesPath = r'/home/aistudio/data'***
2. *打开/home/aistudio/work/cityscapesscripts/paddle/helpers/labels.py文件，**修改61-98行  的labels中trainId项***
3. 运行createTrainIdLabelImgs.py文件
_________________________________
其中createTrainIdLabelImgs.py中的修改是向系统中添加/home/aistudio/work/paddle/cityscapesscripts，/home/aistudio/work/paddle/cityscapesscripts/helpers路径，  定义数据集路径


以上内容中的/home/aistudio/work、/home/aistudio/data根据代码所在目录、数据所在目录进行调整
