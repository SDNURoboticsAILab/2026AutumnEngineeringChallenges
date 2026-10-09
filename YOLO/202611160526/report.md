## 一、记录环境
- 电脑：Windows 11
- Python：3.11.8
- YOLOv8n模型
- 2.14.0+cpu

## 二、数据处理
GitHub上给的数据集，一共943张图，分三个文件夹：
- cola文件夹：可乐的图
- football文件夹：足球的图
- obstacle文件夹：障碍物的图
- 把所有图片约按8:2分了，80%用来训练，20%用来验证：
- 训练集：759张
- 测试集：190张

## 三、标注
- 用Labellmg工具做的标注
- 三个类别编号：
- 0 = cola（可乐）
- 1 = football（足球）
- 2 = obstacle（障碍物）

## 四、训练
- 借助ai进行训练 最终mAP50是0.708
- 三个类别的效果：
- cola：0.721
- football：0.669
- obstacle：0.712

## 五、实验结果
- 模型能够识别出测试集的190张图片中的3种物品
### Level 4：新图片检测
![可乐检测结果](results/level/level%204/kele.png)
![障碍物检测结果](results/level/level%204/zhangai.png)
![足球检测结果](results/level/level%204/zuqiu.png)
- 各Level提交内容位于results文件夹中

## 六、问题与解决过程
### 1.python版本与Labellmg工具冲突
最开始安装Labellmg工具的时候，python版本不匹配，Labellmg工具总是出bug报错闪退，后来我查了相关教程借助ai写的代码修复了好多次更换了适配的环境，才可以正常的的把所有图片标注完
### 2.显卡类型与PyTorch不匹配
电脑显卡不能用GPUban版本的PyTorch，只能用CPU版本的，所以训练速度比较慢
### 3. 标签出错了
刚开始标注好的时候所有标签都写成0了，模型只能识别可乐，后来我对照标注教程重新核对标签，修改了标注文件，模型才能识别三种物品
### 4.不知道GitHub markdown用法
之前没有用过GitHub提交项目，也不会写markdown文档 ，跟着网上教程逐渐学会了仓科创建、文件上传以及markdown大标题 小标题的使用 图片引用等
