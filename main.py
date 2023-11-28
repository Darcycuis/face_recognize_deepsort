import torch, sys, argparse, cv2, os, time
from datetime import datetime
from human_recognize.self_utils.multi_tasks import Counting_Processing
from human_recognize.self_utils.overall_method import Object_Counter, Image_Capture
from human_recognize.deep_sort.configs.parser import get_config
from human_recognize.deep_sort.deep_sort import DeepSort
from human_recognize.self_utils.globals_val import Global
import imutils
import torch.nn as nn
from packaging import version

import numpy as np  # 导入NumPy模块
import insightface  # 导入insightface模块
from Face_recognize.insightface.app import FaceAnalysis  # 从insightface.app中导入FaceAnalysis类
from Face_recognize.insightface.utils import compare
from Face_recognize.insightface.register import find_name, read_csv

# 比较两个包的版本号，码因为pytorch1.10.0以上版本会导致代出现小bug，所以必须判断不同版本的pytorch以增强代码的健壮性
def compare_versions(version1, version2):
    v1 = version.parse(version1)
    v2 = version.parse(version2)
    if v1 > v2:
        return 1
    elif v1 < v2:
        return -1
    else:
        return 0


# 函数来实现自定义的type：
def parse_dict(arg):
    try:
        key, value = arg.split(':')
        return {key: value}
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Use key:value")


def main(yolo5_config, face_args, face_app):
    print("=> 任务开始于: {}".format(datetime.now().strftime('%H:%M:%S')))
    a = time.time()
    class_names = []
    # 加载模型
    if yolo5_config.device != "cpu":
        Model = torch.load(yolo5_config.weights, map_location=lambda storage, loc: storage.cuda(int(yolo5_config.device)))[
        'model'].float().fuse().eval()
    else:
        Model = torch.load(yolo5_config.weights, map_location=torch.device('cpu'))['model'].float().fuse().eval()

    # 本项目最好运行在版本小于等于1.10.0的pytorch上
    best_torch_version = "1.10.0"
    # 当前torch版本
    current_torch_version = str(torch.__version__).split('+')[0]
    print("当前pytorch 版本:", current_torch_version)
    result = compare_versions(best_torch_version, current_torch_version)
    # 如果当前torch版本大于1.10.0，则需要设置一下recompute_scale_factor
    if result == -1:
        for m in Model.modules():
            if isinstance(m, nn.Upsample):
                m.recompute_scale_factor = None
    # 模型能检测的类别['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', ...]
    classnames = Model.module.names if hasattr(Model, 'module') else Model.names
    # 只检测人这一个类别，所以只需要在框上面标识出是person即可
    # 如果想检测其他物体需要在100行附近更改'--classes'的数值，然后在这里把标签改为对应即可

    class_names.append(classnames[0])
    b = time.time()
    print("==> 检测类别: ", class_names)
    print("=> 加载模型, 消耗:{:.2f}s".format(b - a))
    # os.makedirs(yolo5_config.output, exist_ok=True)
    c = time.time()
    # 初始化追踪器deepsort_tracker
    cfg = get_config()
    cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
    deepsort_tracker = DeepSort(cfg.DEEPSORT.REID_CKPT, max_dist=cfg.DEEPSORT.MAX_DIST,
                                min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, max_age=cfg.DEEPSORT.MAX_AGE,
                                n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                use_cuda=True, use_appearence=True)
    # 输入需要检测的图片或图片目录或视频并处理
    mycap = Image_Capture(yolo5_config.input)
    # 实例化计数器, 主要在图片上绘制当前人数以及总人数
    Obj_Counter = Object_Counter(class_names)
    # 总帧数
    total_num = mycap.get_length()
    fps_time = time.time()
    f = 0
    while mycap.ifcontinue():
        f += 1
        if len(yolo5_config.input) <= 1:
            # 从摄像头读入
            ret, img, name = mycap.read()
        else:
            # 从本地视频读入
            ret, img = mycap.read()
        if ret:
            # 开始检测图片中的人
            """
                isCountPresent:
                    True：表示只显示当前人数
                    False：表示显示总人数和当前人数
            """
            # Counting_Processing 函数封装了 yolov5检测，deepsort跟踪计数函数
            result_img = Counting_Processing(img, yolo5_config, face_args, face_app, Model, class_names, deepsort_tracker, Obj_Counter,  isCountPresent = False)
            result_img = imutils.resize(result_img, height=800)
            result_img = cv2.putText(result_img, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                                (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            fps_time = time.time()
            cv2.imshow('video', result_img)
            cv2.waitKey(1)

            if cv2.getWindowProperty('video', cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

        sys.stdout.write("\r=> processing at %d; total: %d" % (mycap.get_index(), total_num))
        sys.stdout.flush()

    print("\n=> 总检测人数:", len(Global.total_person))
    cv2.destroyAllWindows()
    mycap.release()
    print("=> 任务结束于: {}".format(datetime.now().strftime('%H:%M:%S')))


source = "./test8.mp4"
# source = '0'
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    # 视频的路径，默认是本项目中的一个测试视频test.mp4，可自行更改
    parser.add_argument('--input', type=str, default=source,
                        help='test imgs folder or video or camera')  # 输入'0'表示调用电脑默认摄像头
    parser.add_argument('--weights', type=str, default='weights/yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.4, help='IOU threshold for NMS')
    # GPU（0表示设备的默认的显卡）或CPU
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 通过classes来过滤yolo要检测类别，0表示检测人，1表示自行车，更多具体类别数字可以在19行附近打印出来
    parser.add_argument('--classes', default=0, type=int, help='filter by class: --class 0, or --class 0 1 2 3')
    # 是否检测人脸
    parser.add_argument('--detect_face', type=bool, default=True, help='whether to detect faces')
    # 重复检测时间间隔
    parser.add_argument('--gap_time', type=int, default=5, help='Time for repeated detecting')
    # 对异常人员重复检测的时间间隔
    parser.add_argument('--guest_detect_time', type=int, default=2, help='Time for guest detect')
    # 对异常人员的最大检测次数
    parser.add_argument('--guest_Maximum_times', type=int, default=4, help='Maximum detection times for abnormal person')
    yolo5_config = parser.parse_args()

    # insightface 模型参数设置
    # 读取人脸特征
    features_all_path = './Face_recognize/face_database/face_data_all.csv'
    if os.path.exists(features_all_path):
        print(f"File '{features_all_path}' exist.")
        face_features_embedding = np.genfromtxt(features_all_path, delimiter=',')
    mapping_path = './Face_recognize/face_database/mapping_sort.csv'
    if os.path.exists(mapping_path):
        print(f"File '{mapping_path}' exist.")
        map_dict = read_csv(mapping_path)
    assert insightface.__version__ >= '0.3'  # 断言版本不低于0.3
    parser2 = argparse.ArgumentParser(description='insightface app test')  # 创建参数解析器，设置描述为'insightface app test'
    # 通用设置
    parser2.add_argument('--ctx', default=0, type=int,
                        help='ctx id, <0 means using cpu')  # 添加参数'--ctx'，默认值为0，类型为整数，帮助信息为'ctx id, <0 means using cpu'
    parser2.add_argument('--det-size', default=640, type=int,
                        help='detection size')  # 添加参数'--det-size'，默认值为640，类型为整数，帮助信息为'detection size'
    parser2.add_argument('--face_features_embedding', default=face_features_embedding,  nargs='*', type=int,
                         help='All facial features in the facial database')
    parser2.add_argument('--map_dict', default=map_dict, type=parse_dict,
                         help='Detection number with name mapping table')
    face_args = parser2.parse_args()  # 解析参数
    face_app = FaceAnalysis()  # 创建FaceAnalysis实例
    face_app.prepare(ctx_id=face_args.ctx, det_size=(face_args.det_size, face_args.det_size))  # 准备分析器，设置ctx_id和det_size

    main(yolo5_config, face_args, face_app)
