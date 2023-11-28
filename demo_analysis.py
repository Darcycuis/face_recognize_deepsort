import argparse  # 导入参数解析模块
import cv2  # 导入OpenCV模块
import sys  # 导入sys模块
import numpy as np  # 导入NumPy模块
import insightface  # 导入insightface模块
from insightface.app import FaceAnalysis  # 从insightface.app中导入FaceAnalysis类
from insightface.data import get_image as ins_get_image  # 从insightface.data中导入get_image函数
import time
assert insightface.__version__>='0.3'  # 断言版本不低于0.3

parser = argparse.ArgumentParser(description='insightface app test')  # 创建参数解析器，设置描述为'insightface app test'
# 通用设置
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')  # 添加参数'--ctx'，默认值为0，类型为整数，帮助信息为'ctx id, <0 means using cpu'
parser.add_argument('--det-size', default=640, type=int, help='detection size')  # 添加参数'--det-size'，默认值为640，类型为整数，帮助信息为'detection size'
args = parser.parse_args()  # 解析参数

app = FaceAnalysis()  # 创建FaceAnalysis实例
app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))  # 准备分析器，设置ctx_id和det_size
t= time.time()
img = ins_get_image('t1')  # 获取图像't1'
# t= time.time()
faces = app.get(img)  # 识别图像中的人脸
e = time.time()
print("识别人脸：", e-t)
# assert len(faces)==6  # 断言人脸数量为6
rimg = app.draw_on(img, faces)  # 在图像上绘制检测到的人脸
cv2.imwrite("./t1_output.jpg", rimg)  # 将结果图像保存为"t1_output.jpg"

# 然后打印两两人脸之间的相似度
feats = []  # 创建空列表feats
test = []
for face in faces:  # 遍历每个人脸
    feats.append(face.normed_embedding)  # 将人脸的嵌入特征加入feats列表
test.append(faces[0].normed_embedding)
test = np.array(test, dtype=np.float32)
feats = np.array(feats, dtype=np.float32)  # 将feats转换为NumPy数组，数据类型为np.float32
a = time.time()
sims = np.dot(feats, feats.T)  # 计算feats和其转置之间的点积，得到相似度矩阵
b = time.time()
print(sims)  # 输出相似度矩阵
print("用时1：", b-a)

# # 使用landmark_2d_106 计算相似度
# land = []
# for face in faces:
#     land.append(face.landmark_2d_106)
# land = np.array(land, dtype=np.float32)  # 将feats转换为NumPy数组，数据类型为np.float32
# def euclidean_distance(landmarks1, landmarks2):
#     # 计算两组特征点之间的距离
#     distances = np.sqrt(np.sum((landmarks1 - landmarks2)**2, axis=1))
#     # 返回平均距离作为匹配度
#     return np.mean(distances)
# dist_matrix = np.zeros((len(land), len(land)))
# # 计算欧氏距禮以进行人脸比对
# c = time.time()
# for i in range(len(land)):
#     for j in range(len(land)):
#         dist_matrix[i, j] = euclidean_distance(land[i], land[j])
# print("The distance matrix between the faces is:", dist_matrix)
# d = time.time()
# print("用时2：", d-c)