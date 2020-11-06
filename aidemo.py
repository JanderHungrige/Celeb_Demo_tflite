import os
import sys
import gi
import cv2
import numpy as np
from threading import Event, Thread, Lock
from queue import Queue

gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository.GdkPixbuf import Colorspace, Pixbuf
from gi.repository import Gtk, GLib, GObject

#import camerapc as camera
import cameraimx8mp as camera

from ai import Ai


class AiDemo(Gtk.Window):
    def __init__(self, int_event):
        Gtk.Window.__init__(self, title='Celebrity Face Match')

        model_file = 'demo-data/models/tflite/quantized_modelh5-15.tflite'
        embeddings_file = 'demo-data/EMBEDDINGS_quantized_modelh5-15.json'
        self.ai = Ai(os.path.join(sys.path[0], model_file),
                     os.path.join(sys.path[0], embeddings_file),
                     modeltype = 'normal', runtime='tflite',
                     preferred_backend='npu')
        #self.ai.initialize()

        self.cap = camera.get_camera()
        if not self.cap.isOpened():
            raise FileNotFoundError("Failed to open Videodevice")

        # self.set_default_size(1280, 720)
        self.set_position(Gtk.WindowPosition.CENTER)
        #self.fullscreen()
        self.set_border_width(10)
        self.int_event = int_event
        self.key_event = Event()
        self.trigger_event = Event()
        self.contineous = False
        self.connect('key-press-event', self.key_pressed)

        self.image_stream = Gtk.Image()
        self.image_face = Gtk.Image()
        self.image_celeb = Gtk.Image()
        self.you_label = Gtk.Label()
        self.celeb_label = Gtk.Label()
        self.result_label = Gtk.Label()
        self.main_label = Gtk.Label()
        self.switch_label = Gtk.Label()
        self.trigger_btn = Gtk.Button()
        self.trigger_btn.connect('clicked', self.trigger_clicked)
        self.mode_switch = Gtk.Switch()
        self.mode_switch.connect('notify::active', self.mode_switch_action)
        self.mode_switch.set_active(False)
        self.lock_control = Lock()

        self.pic_size = (300, 300)

        self.setup_layout()

        self.face_cascade = cv2.CascadeClassifier(
            'demo-data/lbpcascade_frontalface_improved.xml')

        self.celebs = []
        celeb = cv2.imread('demo-data/danny.jpg')
        celeb = cv2.cvtColor(celeb, cv2.COLOR_BGR2RGB)
        celeb = cv2.resize(celeb, self.pic_size, interpolation=cv2.INTER_CUBIC)
        self.celebs.append(celeb)
        celeb = cv2.imread('demo-data/fairuza.jpg')
        celeb = cv2.cvtColor(celeb, cv2.COLOR_BGR2RGB)
        celeb = cv2.resize(celeb, self.pic_size, interpolation=cv2.INTER_CUBIC)
        self.celebs.append(celeb)
        celeb = cv2.imread('demo-data/richard.jpg')
        celeb = cv2.cvtColor(celeb, cv2.COLOR_BGR2RGB)
        celeb = cv2.resize(celeb, self.pic_size, interpolation=cv2.INTER_CUBIC)
        self.celebs.append(celeb)
        celeb = cv2.imread('demo-data/shirley.jpg')
        celeb = cv2.cvtColor(celeb, cv2.COLOR_BGR2RGB)
        celeb = cv2.resize(celeb, self.pic_size, interpolation=cv2.INTER_CUBIC)
        self.celebs.append(celeb)
        celeb = cv2.imread('demo-data/vin.jpg')
        celeb = cv2.cvtColor(celeb, cv2.COLOR_BGR2RGB)
        celeb = cv2.resize(celeb, self.pic_size, interpolation=cv2.INTER_CUBIC)
        self.celebs.append(celeb)

        self.cam = cv2.imread('demo-data/camera.jpg')
        self.cam = cv2.cvtColor(self.cam, cv2.COLOR_BGR2RGB)
        self.cam = cv2.resize(self.cam, self.pic_size, interpolation=cv2.INTER_CUBIC)

        self.update_face(self.cam)
        self.update_celeb(self.celebs[0])
        self.update_stream(self.celebs[0])

        stream_thread = Thread(target=self.stream)
        faces_thread = Thread(target=self.detect_faces)

        self.image_queue = Queue(maxsize=1)
        self.faces = []
        self.lock_faces = Lock()

        stream_thread.daemon = True
        faces_thread.daemon = True
        stream_thread.start()
        faces_thread.start()

    def setup_layout(self):
        self.main_label.set_markup(
            '<span font="20" font_weight="bold"> Celebrity Face Match </span>'
        )
        self.switch_label.set_markup(
            '<b>Contineous Mode</b>'
        )
        self.result_label.set_markup(
            '<span font="16.0" font_weight="bold">Last Result</span>'
        )
        self.you_label.set_markup(
            '<span font="14.0" font_weight="bold">Your Face</span>'
        )
        self.celeb_label.set_markup(
            '<span font="14.0" font_weight="bold">Your Celebrity</span>'
        )
        self.you_label.set_valign(Gtk.Align.START)
        self.you_label.set_margin_bottom(50)
        self.result_label.set_valign(Gtk.Align.START)
        self.result_label.set_margin_bottom(20)
        self.image_face.set_valign(Gtk.Align.START)
        self.celeb_label.set_valign(Gtk.Align.START)
        self.image_celeb.set_valign(Gtk.Align.START)

        btn_label = Gtk.Label()
        btn_label.set_markup(
            '<span font="14.0" font_weight="bold">Trigger</span>'
        )
        self.trigger_btn.add(btn_label)
        self.trigger_btn.set_size_request(270, 80)
        self.trigger_btn.grab_focus()

        self.mode_switch.set_valign(Gtk.Align.CENTER)
        switch_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=30)
        switch_box.set_halign(Gtk.Align.START)
        switch_box.pack_start(self.switch_label, False, True, 0)
        switch_box.pack_start(self.mode_switch, False, True, 0)

        trigger_box = Gtk.Box(spacing=5)
        trigger_box.set_homogeneous(False)
        trigger_box.set_valign(Gtk.Align.END)
        trigger_box.set_halign(Gtk.Align.START)
        trigger_box.pack_start(switch_box, False, True, 0)
        trigger_box.pack_start(self.trigger_btn, False, True, 300)

        stream_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        stream_box.set_valign(Gtk.Align.START)
        stream_box.set_halign(Gtk.Align.CENTER)
        stream_box.pack_start(self.image_stream, False, True, 0)
        stream_box.pack_start(trigger_box, False, True, 0)

        picture_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        picture_box.set_homogeneous(False)
        picture_box.pack_start(self.result_label, False, True, 0)
        picture_box.pack_start(self.image_face, False, True, 0)
        picture_box.pack_start(self.you_label, False, True, 0)
        picture_box.pack_start(self.image_celeb, False, True, 0)
        picture_box.pack_start(self.celeb_label, False, True, 0)

        content_box = Gtk.Box(spacing=10)
        content_box.pack_start(stream_box, True, True, 0)
        content_box.pack_start(picture_box, True, True, 0)
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_box.pack_start(self.main_label, True, True, 0)
        main_box.pack_start(content_box, True, True, 0)
        self.add(main_box)

    def stream(self):
        count = 0
        while True:
            if self.int_event.is_set():
                break

            ret, frame = self.cap.read()
            if ret == 0:
                print('No Frame')
                continue

            frame = camera.color_convert(frame)

            if self.image_queue.full():
                self.image_queue.get()

            self.image_queue.put(frame.copy())

            with self.lock_faces:
                if self.faces:
                    (x, y, w, h) = self.faces[0]
                    self.faces = []
                    p = 35
                    if (x - p < 0 or y - p < 0 or
                          x + w + p > np.shape(frame)[1] or
                          y + h + p > np.shape(frame)[0]):

                        face = self.cam
                    else:
                        frame = cv2.rectangle(frame, (x-p, y-p+2),
                                              (x+w+p, y+h+p+2),
                                              (0, 255, 0), 2)
                        face = frame[y-p+4:y+h+p, x-p+4:x+w+p]
                        face = cv2.resize(face, self.pic_size,
                                          interpolation=cv2.INTER_CUBIC)

                else:
                    face = self.cam

            GLib.idle_add(self.update_face, face,
                          priority=GLib.PRIORITY_DEFAULT_IDLE)

            GLib.idle_add(self.update_stream, frame,
                          priority=GLib.PRIORITY_DEFAULT_IDLE)

            if self.key_event.is_set():
                new_frame = frame.copy()
                GLib.idle_add(self.update_face, new_frame,
                              priority=GLib.PRIORITY_DEFAULT_IDLE)
                self.key_event.clear()

            GLib.idle_add(self.update_celeb, self.celebs[count],
                          priority=GLib.PRIORITY_DEFAULT_IDLE)

            if count < 4:
                count += 1
            else:
                count = 0

    def detect_faces(self):
        while True:
            if self.int_event.is_set():
                break

            frame = self.image_queue.get()

            scale = 4
            (h, w, c) = np.shape(frame)
            frame = cv2.resize(frame, (int(w/scale), int(h/scale)))
            faces = self.face_cascade.detectMultiScale(frame,
                                                       scaleFactor=1.2,
                                                       minNeighbors=5)

            if len(faces) == 0:
                with self.lock_faces:
                    self.faces = []
                continue

            frame_center = np.shape(frame)[1] / 2
            face_centers = []
            # Find face which is closest to the center
            for (x, y, w, h) in faces:
                face_centers = np.append(face_centers, (x + w / 2))

            center_face_idx = (np.abs(face_centers - frame_center)).argmin()
            # Extract center face from frame
            (x, y, w, h) = faces[center_face_idx]
            (x, y, w, h) = (x * scale, y * scale, w * scale, h * scale)

            with self.lock_faces:
                self.faces.append((x, y, w, h))

    def key_pressed(self, widget, key):
        print('KEY pressed')
        print(key.keyval)
        self.key_event.set()
        return False

    def trigger_clicked(self, button, gparam):
        self.trigger_event.set()

    def mode_switch_action(self, switch, gparam):
        if switch.get_active():
            active = True
            self.trigger_btn.set_sensitive(False)
        else:
            active = False
            self.trigger_btn.set_sensitive(True)

        with self.lock_control:
            self.contineous = active

    def update_stream(self, frame):
        frame = cv2.resize(frame, (1280, 800))
        height, width = frame.shape[:2]
        arr = np.ndarray.tostring(frame)
        pixbuf = Pixbuf.new_from_data(arr, Colorspace.RGB, False, 8,
                                      width, height, width * 3, None, None)
        self.image_stream.set_from_pixbuf(pixbuf)
        return False

    def update_face(self, face):
        height, width = face.shape[:2]
        arr = np.ndarray.tostring(face)
        pixbuf = Pixbuf.new_from_data(arr, Colorspace.RGB, False, 8,
                                      width, height, width * 3, None, None)
        self.image_face.set_from_pixbuf(pixbuf)
        return False

    def update_celeb(self, celeb):
        height, width = celeb.shape[:2]
        arr = np.ndarray.tostring(celeb)
        pixbuf = Pixbuf.new_from_data(arr, Colorspace.RGB, False, 8,
                                      width, height, width * 3, None, None)
        self.image_celeb.set_from_pixbuf(pixbuf)
        return False


def main(args):
    int_event = Event()
    window = AiDemo(int_event)
    window.connect('delete-event', Gtk.main_quit)
    window.show_all()

    try:
        Gtk.main()
    except KeyboardInterrupt:
        int_event.set()


if __name__ == '__main__':
    main(sys.argv)
