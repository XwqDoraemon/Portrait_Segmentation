import cv2
from PIL import Image, ExifTags
import numpy as np
import traceback

path_1 = 'butongqibian_QR-01326.jpg'
path_2 = 'lot_image002.jpg'


def get_angle(path_img):
    """
    返回此PIL image，应当逆时针旋转的角度
    https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageOps.html?highlight=orientation#
    https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image
    :param path_img:
    :return:
    """
    try:
        image = Image.open(path_img)
        # 获取orientation的key
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            angle = 180
            # image = image.rotate(180, expand=True)  # 逆时针旋转180度
        elif exif[orientation] == 6:
            angle = 270
            # image = image.rotate(270, expand=True)  # 逆时针旋转270度
        elif exif[orientation] == 8:
            angle = 90
            # image = image.rotate(90, expand=True)   # 逆时针旋转90度
        else:
            angle = 0
        return angle
    except:
        return 0
        # traceback.print_exc()


def cvt_img(path):
    img = cv2.imread(path)
    print("cv2：", img.shape)  # H, W, C
    img2 = Image.open(path)
    print("pil：", np.array(img2).shape)  # 对于包含很多元信息的图片，不确定PIL是何种机制导致H，W的调换
    print("{}\n当前PIL Image需要逆时针旋转{}度，才是原始方向".format(path, get_angle(path)))
    # img2.show()


if __name__ == "__main__":
    # cvt_img(path_1)
    cvt_img(path_2)
