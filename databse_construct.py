import argparse  # 导入参数解析模块
import os.path

import cv2  # 导入OpenCV模块
import numpy as np  # 导入NumPy模块
import json
import base64
import socket
from human_recognize.self_utils.socket_method import get_database, send_features_data
from Face_recognize.insightface.app import FaceAnalysis
import multiprocessing


def mean_feature_fusion(features):
    """
    人脸特征进行平均值融合
    Args:
        features: [[特征1],[特征2],[[特征....],.....]
    """
    mean_feature = np.mean(features, axis=0)
    return mean_feature


def features_construct(app, result_queue, result_queue1):
    """
    从 result_queue 队列中取出，从服务器获取的人脸数据库信息(姓名，图片)  {”name“: 姓名, "image": 图片}
    使用app.get 检测人脸特征 并存放如result_queue1 以待发送
    发送数据格式  {”name“: 姓名, "feature": 人脸特征}
    Args:
        app:
        result_queue:
        result_queue1:
    """
    while True:
        while not result_queue.empty():
            face_data = result_queue.get()
            print("face_data:", face_data)
            face_data = json.loads(face_data)
            # print("face_data:",face_data)
            name = face_data['name']
            # print("name:", name)
            image = face_data['image']
            # # 解码图像
            # img_bytes = base64.b64decode(image)
            # # 将字节数据转换为NumPy数组
            # image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            features = []
            data_dict = {}
            for img in image:
                img_bytes = base64.b64decode(img)
                # 将字节数据转换为NumPy数组
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                # im_name = name + '.jpg'
                # path = os.path.join('./images', im_name)
                # cv2.imwrite(path, img)
                face_all = app.get(img)
                for face_single in face_all:  # 遍历每个人脸
                    features.append(face_single.normed_embedding)  # 将人脸的嵌入特征加入features列表
            feature = mean_feature_fusion(features)
            file_path = './feature.csv'
            with open(file_path, 'a', newline='') as file:
                np.savetxt(file, [feature], delimiter=',')
            data_dict['name'] = name
            data_dict['feature'] = feature.tolist()
            result_queue1.put(data_dict)


def database_features_construct(app, result_queue, result_queue1):
    # 服务器地址和端口
    server_address = ('192.168.2.107', 10000)
    server_address2 = ('192.168.2.107', 10001)
    # 创建一个TCP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接到服务器
    client_socket.connect(server_address)
    print(f"Connected to server at {server_address}")
    client_socket2.connect(server_address2)
    print(f"Connected to server at {server_address2}")

    # # 发送消息给服务器
    # message_to_send = "Hello, server! This is the client."
    # client_socket.send(message_to_send.encode('utf-8'))
    #
    # # 接收服务器消息
    # data = client_socket.recv(1024)
    # print(f"Received data from server: {data.decode('utf-8')}")

    get_database_process = multiprocessing.Process(target=get_database, args=(client_socket, result_queue))
    features_construct_process = multiprocessing.Process(target=features_construct, args=(app, result_queue, result_queue1))
    send_features_data_process = multiprocessing.Process(target=send_features_data, args=(client_socket2, result_queue1))

    # 启动进程
    get_database_process.start()
    features_construct_process.start()
    send_features_data_process.start()
    # 等待两个进程结束
    get_database_process.join()
    features_construct_process.join()
    send_features_data_process.join()
    # 关闭连接
    client_socket.close()


if __name__ == "__main__":
    parser2 = argparse.ArgumentParser(description='insightface app test')  # 创建参数解析器，设置描述为'insightface app test'
    # 通用设置
    parser2.add_argument('--ctx', default=0, type=int,
                         help='ctx id, <0 means using cpu')  # 添加参数'--ctx'，默认值为0，类型为整数，帮助信息为'ctx id, <0 means using cpu'
    parser2.add_argument('--det-size', default=640, type=int,
                         help='detection size')  # 添加参数'--det-size'，默认值为640，类型为整数，帮助信息为'detection size'
    face_args = parser2.parse_args()  # 解析参数
    face_app = FaceAnalysis()  # 创建FaceAnalysis实例
    face_app.prepare(ctx_id=face_args.ctx, det_size=(face_args.det_size, face_args.det_size))  # 准备分析器，设置ctx_id和det_size
    result_queue = multiprocessing.Queue()  # 多进程队列
    result_queue1 = multiprocessing.Queue()

    database_features_construct(face_app, result_queue, result_queue1)























