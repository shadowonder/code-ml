# https://digi.bib.uni-mannheim.de/tesseract/
# 或者直接从github中获取 https://github.com/UB-Mannheim/tesseract/wiki
# 如果需要使用tesseract的话, 需要安装到cmd
# 配置环境变量如E:\Program Files (x86)\Tesseract-OCR
# tesseract -v进行测试
# tesseract XXX.png 得到结果 
# pip install pytesseract
# 如果实在无法识别文件, 可以进入到安装路径手动修改一下命令D:\\anaconda\\lib\\site-packges\\pytesseract\\pytesseract.py
# tesseract_cmd 修改为绝对路径即可 D:\\tesseract\\tesseract.exe
import os

from PIL import Image
import cv2
import pytesseract

preprocess = 'blur'  # thresh

image = cv2.imread('E:\Workspace\ml\code-ml\ml\python\computer-vision\opencv\examples\scan.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

if preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)

filename = "{}.png".format(os.getpid())  # 保存扫描结果
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename))
print(text)  # 文本
os.remove(filename)

cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
