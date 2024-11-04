import torch

from L1_2 import batch

#a =torch.ones(3)
#print(a)
#print(a[1],a[1].item())
# 输出：tensor([1., 1., 1.])
#      tensor(1.) 1.0
print("定义张量")
points = torch.tensor([[1,4],[5,3],[2,1]]) # 注意要加两层[]
print(points, points.shape, points.size()) # 在获取张量形状时，没有功能上差别。不同在于 size() 可以接受维度索引作为参数，而 shape 不能
print(points.size(0), points.size(1)) # 输出分别为3，2
print(points[0,1], points[0,1].item())
print(points[0]) # 这里不能用points[0].item()，.item() 常用于提取单个值
# 张量索引
print("-------------------------------------------------------------")
print("张量索引")
print(points[1:]) # 1表示从第二行开始，:表示到结尾
print(points[1:,:]) # ,分隔维度，:表示到结尾
print(points[1:,0]) # 第一行之后所有行，第一列
print(points[None], points.shape) #points[None]增加大小为1的维度，但是不在原数据上修改，points仍然是torch.Size([3, 2])
# 广播特性
print("-------------------------------------------------------------")
print("广播特性")
img_t = torch.randn(3,3,3)
print("img_t是:",img_t)
weights = torch.tensor([0.2126, 0.7152, 0.0722])
batch_t = torch.randn(2,3,3,3) # 加入了batch_size
print("batch_t是:",batch_t)
img_gray_naive = img_t.mean(-3)
# mean(-3) 表示对 img_t 的第 -3 个维度求平均值。
# 对应的是颜色通道的维度 (即第 0 维)。通过在这个维度上求平均值，可以将 RGB 图像转换为灰度图像。
# 所以，img_gray_naive 是一个形状为 (3, 3) 的张量，即对应一个灰度图像。下面同理。
print(img_gray_naive.shape,img_gray_naive)
batch_gray_naive = batch_t.mean(-3)
print(batch_gray_naive.shape,batch_gray_naive) # batch_gray_naive 的形状会变成 (2, 3, 3)，表示 2 张 3x3 的灰度图像。

unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)
print(unsqueezed_weights.shape, unsqueezed_weights) #使用unsqueeze_(-1)直接在原张量上修改形状
# 注意所有以下划线_结尾的都是对Tensor对象的方法，也就是直接在原tensor上操作
print("此时unsqueezed_weights形状为:",unsqueezed_weights.shape)
batch_weights = (batch_t * unsqueezed_weights)
print("原来batch_t形状为:",batch_weights.shape,"与unsqueezed_weights相乘，unsqueezed_weights经过(3,1,1),(1,3,1,1),(2,3,5,5)变化。",
    "成功与batch_t相乘，这就是PyTorch的广播特性。")
print("得到batch_weights形状为：",batch_weights.shape)

print("-------------------------------------------------------------")
print("张量命名")
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names = ['channels']) # 第1维度命名为‘channels’
print(weights_named)
img_named = img_t.refine_names(..., 'channels', 'rows', 'columns') # 命名三个维度，...表示可以省略前面的维度（也就是命名倒数三个维度）
print(img_named.shape, img_named)
weights_alighed = weights_named.align_as(img_named) # 将原来（3）的tensor按照其它tensor的命名方式拓展到（3,1,1),对应维度也同样命名
print(weights_alighed.shape, weights_alighed)
gray_named = (img_named * weights_alighed).sum('channels') # 在‘channels’维度上求和
print(gray_named.shape, gray_named)