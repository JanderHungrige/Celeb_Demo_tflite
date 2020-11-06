import os
import cv2
import subprocess
import fcntl
import v4l2_2


def get_camera():
    videodev = 'video0'
    buildinfo = cv2.getBuildInformation()

    if buildinfo.find('GStreamer') < 0:
        print('no GStreamer support in OpenCV')
        exit(0)

    failed = False
    # Make sure the camera is in a defined state
    cmd = 'media-ctl -V "31:0[fmt:SGRBG8_1X8/1280x800 (4,4)/1280x800]"'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('media-ctl failed: {}'.format(ret))
        failed = True
    cmd = 'media-ctl -V "22:0[fmt:SGRBG8_1X8/1280x800]"'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('media-ctl failed: {}'.format(ret))
        failed = True
    cmd = 'v4l2-ctl -d0 -v width=1280,height=800,pixelformat=GRBG'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('v4l2-ctl failed: {}'.format(ret))
        failed = True
    cmd = 'v4l2-ctl -d0 -c vertical_flip=1'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('v4l2-ctl failed: {}'.format(ret))
        failed = True

    if failed:
        vd = os.open('/dev/video0', os.O_RDWR | os.O_NONBLOCK, 0)
        btype = v4l2_2.v4l2_buf_type(v4l2_2.V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
        fcntl.ioctl(vd, v4l2_2.VIDIOC_STREAMOFF, btype)
        os.close(vd)

    pipeline = 'v4l2src device=/dev/{video} ! appsink'.format(video=videodev)
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


def color_convert(frame):
    monochrome = False
    if not monochrome:
        return cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2RGB)
    else:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
