import json
from torchvision import models #引入models
from torchvision import transforms #引入处理图片的函数
from PIL import Image #用于提取图片
import torch
#print(dir(models))
alexnet = models.AlexNet() # alexnet得到一个没有预训练参数，但是模型结构一样的实例
resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
# resnet得到一个在imagenet上有预训练参数的实例
#print(resnet)
# 数据预处理过程
preprocess = transforms.Compose([
    transforms.Resize((256, 256)), # 图片行宽转换为256*256
    transforms.CenterCrop(224), # 围绕中心进行224*224裁剪
    transforms.ToTensor(), # 转换为tensor数据类型
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) # 对RGB三个通道进行归一化
])
img = Image.open('/Users/fanjialin/Desktop/NJU_scientific research/复现/imagenette2/val_without_label/ILSVRC2012_val_00000732.JPEG')
#img.show()
img_t = preprocess(img) # 预处理图片
batch = img_t.unsqueeze(0) # 在img_t的0维添加一个新的维度，生成包含单个图像的“批量”张量，用于模型输入
print(img_t.shape) # img_t.unsqueeze(0)没有在原数据上修改
resnet.eval() # 将模型置为eval模式
output = resnet(batch)
print(output.size()) #输出为torch.Size([1, 1000])，1行100列，每列表示一种类别的参数
with open('/Users/fanjialin/Desktop/NJU_scientific research/复现/imagenet_class_index.json') as f:
    label_map = json.load(f) # 读取标签
_, index = torch.max(output, 1) # _,是占位符，max返回两个值，一个是最大值本身，另外一个是最大值索引，index存储索引
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100 # 通过softmax得到置信度，*100得到百分比
label_key = str(index[0].item())  # 将索引转换为字符串，以便查找label_map
label_name = label_map[label_key][1]  # 获取类别名称（原来的字典是这样的：'0': ['n01440764', 'tench']）
print(label_name, percentage[index[0]].item()) # index是一个一维tensor张量，需要index[0]获取具体数值
# .item() 方法的作用是将张量中的单个数值提取出来，并将其转换为 Python 的原生数据类型（如 float 或 int）