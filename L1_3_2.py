import torch

print("张量的dtype")
double_points = torch.tensor([1,2,3], dtype = torch.double)
float_points = torch.tensor([1,2,3], dtype = torch.float) # 默认为32位浮点精度
short_points = torch.ones(3, 2, dtype = torch.short)
print(double_points)
print(float_points)
print(short_points)

print("-------------------------------------------------------------")
print("张量API（请探索官方文档）")
a = torch.ones(3,2)
a_t = torch.transpose(a,0,1)
print(a)
print(a_t)
# transpose函数在高维空间内可以理解成对索引的调换，比如原来一个（3，2，2）tensor索引a[0][1][2]=1
# 经过torch.transpose(a,0,1)后变成索引a[1][0][2]=1，其余索引类似，这样就不那么抽象了
# 对图像像素进行transpose后不会影响图像的样子（相当于改一种顺序渲染图像）

print("-------------------------------------------------------------")
print("张量存储空间")
points = torch.tensor([[4,1],[5,3],[2,1]])
points_t = torch.transpose(points,0,1)
print(points)
print(points_t)
print(id(points.storage())==id(points_t.storage())) # 共享一片存储区域，只是多建立了一个points_t的Tensor实例，在形状和步长上不同
print("points的步长：",points.stride(),"points_t的步长：",points_t.stride())
# 步长指当索引在每个维度都增加1时在存储区域里跳过的元素数量
# 值的注意的是这里points_t的形状是（3，2），但是它的步长不是（3.1）而是（1，2），这是因为它仍在使用points存储空间，按照这个存储空间找步长
print("连续张量")
print("points_t张量是不是连续张量：",points_t.is_contiguous())
print("points_t存储空间：",points_t.storage())
points_t_cont = points_t.contiguous() # contiguous函数额外再生成一个存储空间给points_t_cont，不在原来的存储空间上修改
print("points_t_cont存储空间：",points_t_cont.storage())
print("points_t_cont的步长：",points_t_cont.stride())

print("-------------------------------------------------------------")
print("张量存储设备")
# 请注意，以下都是适用于macOS的mps
points_mps = points.to(device='mps') # 分配一个新张量，不是原来那个
print("points_mps在mps上运行：",points_mps.device)
points_mps + 4
points_cpu = points_mps.cpu() # 回到cpu要额外写一下
