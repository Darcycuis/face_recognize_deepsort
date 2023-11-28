
from Face_recognize.insightface.utils import compare
from Face_recognize.insightface.register import find_name

def detect_face( face_features_embedding, dict, feature):
    """
    输入检测图片，以及人脸数据库总特征，返回检测出的姓名
    Args:
        image: 输入人脸图片
        ls:  数组用来存储图片
        face_features_embedding:  人脸数据库总特征
        dict:  员工序号姓名映射词典
        face_app:实例化检测模型
    """
    index = compare(feature, face_features_embedding)
    # 通过映射表查找到人脸特征对应姓名
    detect_name = find_name(index, dict)
    return detect_name

