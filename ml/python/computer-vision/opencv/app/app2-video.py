import cv2

"""
1. 视频的基本操作
在cv中, 图像就是多帧的图片集合. 在学习中, 我们需要将视频的每一帧进行计算
"""
vc = cv2.VideoCapture('E:\Workspace\ml\code-ml\ml\python\computer-vision\data\opencv.mp4')
print(vc)  # <VideoCapture 000001C13D647B10>

# 对视频进行读取, isOpened表示是否可以打开, 如果打开的话 是否可以播放等等
# 然后对于每一帧进行获取. 这里的vc可以看做是一个迭代器, 对其进行迭代
if vc.isOpened():
    # read()方法表示读取一帧进行迭代, 返回两个值, 第一个值表示的是是否读取成功, 第二个值表示的是读取的结果
    # 读取出来的frame图像就是image图像
    open, frame = vc.read()
else:
    open = False

while open:
    ret, frame = vc.read()
    if frame is None:  # 当图片结束的时候, 读取的帧就是空帧, 当读取到空的时候跳出循环
        break
    if ret == True:  # 当读取成功的时候我们进行展示
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 将图像进行灰度处理, 获得灰度处理的图像变量"gray"
        cv2.imshow('result', gray)  # 展示这个图像
        # 当这个图像等待0.01秒, 进入下一个图像,或者退出字节为27,这里表示的是退出按键为esc的时候退出循环
        if cv2.waitKey(10) & 0xFF == 27:
            break
vc.release()  # 释放视频
cv2.destroyAllWindows()
