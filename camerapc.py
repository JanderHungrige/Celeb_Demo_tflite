import cv2


def get_camera():
    return cv2.VideoCapture(1)


def color_convert(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)