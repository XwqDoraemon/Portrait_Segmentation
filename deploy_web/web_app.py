# -*- coding:utf-8 -*-
"""
# @file name  : decorator_demo.py
# @author     : XueWQ
# @date       : 2021-06-01
# @brief      : 对图片进行分割推理
"""
from flask import Flask, render_template, request
import time
import os
import cv2
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1,os.path.join(BASE_DIR, '..'))
from tools.predictor import Predictor
app = Flask(__name__)

# 初始化


def save_img(file, out_dir):
    time_stamp = str(time.time())
    file_name = time_stamp + file.filename
    path_to_img = os.path.join(out_dir, file_name)
    file.save(path_to_img)
    return path_to_img, file_name

def gen_html(img, img_bgr, file_name, out_dir):
    matting_name = os.path.splitext(file_name)[0] + "_matting.jpg"
    resize_name = os.path.splitext(file_name)[0] + "_resize.jpg"
    path_matting = os.path.join(out_dir, matting_name)
    path_resize = os.path.join(out_dir, resize_name)
    # cv2.imwrite(path_matting, img)   # 无法保存中文路径
    # cv2.imwrite(path_resize, img_bgr) # 无法保存中文路径
    cv2.imencode('.jpg', img)[1].tofile(path_matting)
    cv2.imencode('.jpg', img_bgr)[1].tofile(path_resize)

    show_info = {"path_resize": path_resize,
                 "path_matting": path_matting,
                 "width": img_bgr.shape[1],
                 "height": img_bgr.shape[0]}
    name_html = "matting_result_{}.html".format(file_name)
    path_html = os.path.join(BASEDIR, "templates", name_html)

    html_template(show_info, path_html)
    return name_html


def html_template(show_info, path_html):
    html_string_start = """
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <style>
    #left-bar {
      position: fixed;
      display: table-cell;
      top: 100;
      bottom: 10;
      left: 10;
      width: 50%;
      overflow-y: auto;
    }
    #right-bar {
      position: fixed;
      display: table-cell;
      top: 100;
      bottom: 10;
      right: 10;
      width: 35%;
      overflow-y: auto;
    }
    </style>
    <body>
    """

    html_string_end = """

    </body>
    </html>

    """

    path_resize = "../static" + show_info["path_resize"].split("static")[-1]
    img_resize_html = """<div id= "left-bar" > 
    <picture> <img src="{}" height="{}" width="{}"> </picture> <br>原始图片<br>""".format(
        path_resize, show_info["height"], show_info["width"])

    path_matting = "../static" + show_info["path_matting"].split("static")[-1]
    img_matting_html = """<div id= "right-bar" > 
    <picture> <img src="{}" height="{}" width="{}"> </picture><br>效果图片<br>""".format(
        path_matting, show_info["height"], show_info["width"])

    file_content = html_string_start + img_resize_html + img_matting_html + html_string_end
    with open(path_html, 'w', encoding="utf-8") as f:
        f.write(file_content)


# 定义该url接收get和post请求， 可到route的add_url_rule函数中看到默认是get请求
@app.route("/", methods=["GET", "POST"])
def func():
    # request 就是一个请求对象，用户提交的请求信息都在request中
    if request.method == "POST":
        try:
            # step1: 接收传入的图片
            f = request.files['imgfile']
            path_img, file_name = save_img(f, upload_dir)
            # step2：推理
            img_t, img_bgr = predictor.preprocess(path_img, in_size=in_size, keep_ratio=True)
            _, pred_mask = predictor.predict(img_t)
            img_matting = predictor.postprocess(img_bgr, pred_mask, color="w")
            # step3: 生成用于展示的html
            name_html = gen_html(img_matting, img_bgr, file_name, upload_dir)
            return render_template(name_html)
        except Exception as e:
            return f"{e}, Please try it again!"
    else:
        return render_template("upload.html")

if __name__ == '__main__':
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    upload_dir = os.path.join(BASE_DIR, "static", "upload_img")
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    path_checkpoint = "/hy-tmp/output_seg/checkpoint_best.pkl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = Predictor(path_checkpoint, device=device)
    in_size = 512

    app.run()

    # 允许外部访问，但无公网IP，仅局域网内其他主机可访问，如同wifi下的设备，本机IP可通过ipconfig命令查看， mac通过ipconfig /a
    # app.run(host="0.0.0.0", port=80)
