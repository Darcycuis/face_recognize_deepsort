import argparse  # 导入参数解析模块
import cv2  # 导入OpenCV模块
import numpy as np  # 导入NumPy模块
import insightface  # 导入insightface模块
from insightface.app import FaceAnalysis  # 从insightface.app中导入FaceAnalysis类
import time
import os
from overall_method import Image_Capture
from insightface.utils import compare
from insightface.register import find_name, read_csv
assert insightface.__version__>='0.3'  # 断言版本不低于0.3
parser = argparse.ArgumentParser(description='insightface app test')  # 创建参数解析器，设置描述为'insightface app test'
# 通用设置
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')  # 添加参数'--ctx'，默认值为0，类型为整数，帮助信息为'ctx id, <0 means using cpu'
parser.add_argument('--det-size', default=640, type=int, help='detection size')  # 添加参数'--det-size'，默认值为640，类型为整数，帮助信息为'detection size'
args = parser.parse_args()  # 解析参数

app = FaceAnalysis()  # 创建FaceAnalysis实例
app.prepare(ctx_id=args.ctx, det_size=(args.det_size, args.det_size))  # 准备分析器，设置ctx_id和det_size

source = "1"
# source = './ti.jpg'
source = "./insightface/data/images/test22.mp4"
mycap = Image_Capture(source)
f = 0
# 读取人脸特征
features_all_path = './face_database/face_data_all.csv'
if os.path.exists(features_all_path):
    print(f"File '{features_all_path}' exist.")
    face_features_embedding = np.genfromtxt(features_all_path, delimiter=',')
mapping_path = './face_database/mapping_sort.csv'
dict = read_csv(mapping_path)
while mycap.ifcontinue():
    f += 1
    fps_time = time.time()
    if len(source) <= 1:
        # 从摄像头读入
        ret, img, name = mycap.read()
    else:
        # 从本地视频读入
        ret, img = mycap.read()
    if ret:
        t = time.time()
        faces = app.get(img)  # 识别图像中的人脸
        e = time.time()
        print("识别人脸：", e-t)
        names = []
        for face in faces:  # 遍历每个人脸
            index = compare(face.normed_embedding, face_features_embedding)
            print("index:", index)
            detect_name = find_name(index, dict)
            names.append(detect_name)
        # assert len(faces)==6  # 断言人脸数量为6
        rimg = app.draw_name(img, faces, names)  # 在图像上绘制检测到的人脸
        # cv2.imwrite("./t-_output.jpg", rimg)  # 将结果图像保存为"t1_output.jpg"
        result_img = cv2.putText(rimg, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                                 (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        rimg = cv2.resize(result_img, (1200, 700))
        cv2.imshow('video', rimg)
        cv2.waitKey(1)
        if cv2.getWindowProperty('video', cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
