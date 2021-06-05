import tkinter
import tkinter.messagebox
from tkinter.filedialog import *

import cv2
import numpy as np
import torch
from PIL import ImageTk, Image, ImageGrab
import win32con
import win32gui
import win32print

from win32api import GetSystemMetrics
from img2inchi_transformer import Img2InchiTransformerModel
from pkg.utils.general import Config
from pkg.utils.vocab import vocab as vocabulary


def get_real_resolution():
    """获取真实的分辨率"""
    hDC = win32gui.GetDC(0)
    # 横向分辨率
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    # 纵向分辨率
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    return w, h


def get_screen_size():
    """获取缩放后的分辨率"""
    w = GetSystemMetrics(0)
    h = GetSystemMetrics(1)
    return w, h


real_resolution = get_real_resolution()
screen_size = get_screen_size()
screen_scale_rate = round(real_resolution[0] / screen_size[0], 2)
target_image = None


class window:
    def __init__(self):
        self.win = tkinter.Tk()
        self.i = 0

        model_config = os.getcwd() + '/model_weights/tfe3d6/export_config.yaml'
        config = Config(model_config)
        self.my_vocab = vocabulary(root=config.path_train_root, vocab_dir=config.vocab_dir)
        self.model = Img2InchiTransformerModel(config, output_dir='', vocab=self.my_vocab, need_output=False)
        self.model.build_pred(os.getcwd() + '/model_weights/tfe3d6/model.ckpt', config=config)

        self.win.title('img2inchi')
        self.win.geometry('1000x1000')
        self.imglabel = None
        self.inchi = None
        self.text = tkinter.Text(self.win, width=85, height=7)
        self.text.place(x=200, y=650)
        B1 = tkinter.Button(self.win, text="Import Image", font=('Arial', 12), command=lambda: self.importimg(),
                            width=15, height=1)
        B1.place(x=100, y=10)
        B2 = tkinter.Button(self.win, text="Import Image", font=('Arial', 12), command=lambda: self.screenshot(),
                            width=15, height=1)
        B2.place(x=250, y=10)
        B3 = tkinter.Button(self.win, text="Begin Transform", font=('Arial', 12), command=lambda: self.transform(),
                            width=15, height=1)
        B3.place(x=600, y=10)
        B4 = tkinter.Button(self.win, text="next attention image", font=('Arial', 12), command=lambda: self.nextimg(),
                            width=15, height=1)
        B4.place(x=400, y=750)
        B5 = tkinter.Button(self.win, text="image preprocess", font=('Arial', 12), command=lambda: self.imgprocess(),
                            width=15, height=1)
        B5.place(x=550, y=10)

        self.win.mainloop()

    def importimg(self):
        global target_image
        img_dir = askopenfilenames()
        if self.imglabel:
            self.imglabel.destroy()
        target_image = cv2.imread(img_dir[0].replace('/', '\\'))
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        if target_image is not None:
            if target_image.shape != (512, 256):
                self.text.insert("end", "The size of current image is " + str(target_image.shape) + "\n")
            photo = ImageTk.PhotoImage(image=Image.fromarray(target_image))
            self.imglabel = tkinter.Label(self.win, image=photo)
            self.imglabel.place(x=250, y=50)
            self.text.insert("end", "Import image succeed!\n")
            self.flag = 0
            self.win.mainloop()

    def imgprocess(self):
        global target_image
        if target_image is None:
            self.text.insert("end", "No image imported!\n")
            return
        if self.flag == 2:
            self.text.insert("end", "image has been processed!\n")
            return
        if self.imglabel:
            self.imglabel.destroy()
        h, w = target_image.shape
        if h > w:
            target_image = np.rot90(target_image)
            h, w = target_image.shape
        pad_h, pad_v = 0, 0
        hw_ratio = (h / w) - (256 / 512)
        if hw_ratio < 0:
            pad_h = int(abs(hw_ratio) * w / 2)
        else:
            wh_ratio = (w / h) - (512 / 256)
            pad_v = int(abs(wh_ratio) * h // 2)
        target_image = np.pad(target_image, [(pad_h, pad_h), (pad_v, pad_v)], mode='constant', constant_values=255)
        target_image = cv2.resize(target_image, (512, 256), interpolation=cv2.INTER_LANCZOS4)
        target_image = (target_image / target_image.max() * 255).astype(np.uint8)
        target_image = cv2.bitwise_not(target_image)
        target_image = cv2.morphologyEx(target_image, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        photo = ImageTk.PhotoImage(image=Image.fromarray(target_image))
        self.imglabel = tkinter.Label(self.win, image=photo)
        self.imglabel.place(x=250, y=50)
        self.text.insert("end", "process image succeed!\n")
        self.flag = 1
        self.win.mainloop()

    def transform(self):
        if self.flag == 0:
            self.text.insert("end", "image needs to be processed first!\n")
            return
        self.flag = 2
        global target_image
        img = torch.from_numpy(target_image).float()
        img = img.repeat(1, 3, 1, 1)
        result = self.model.predict(img, mode="beam")
        seq = self.my_vocab.decode(result[0])
        self.text.insert("end", seq)
        self.text.insert("end", "\n")
        attn = self.model.get_attention(img, result)
        attn = attn.cpu().numpy()
        attn = attn[0, 0]
        nhead, self.lenth, h, w = attn.shape
        raw_img = target_image
        assert nhead == 8
        attn = np.reshape(np.transpose(attn, (1, 0, 2, 3)), newshape=(self.lenth, 4, 2, h, w))
        attn = np.reshape(np.transpose(attn, (0, 1, 3, 2, 4)), newshape=(self.lenth, 4 * h, 2 * w))
        imgh, imgw = raw_img.shape
        imgh = int(imgh / 1.6)
        imgw = int(imgw / 1.6)
        raw_img = cv2.resize(raw_img, (imgw, imgh), interpolation=cv2.INTER_LANCZOS4)
        raw_img = np.reshape(np.transpose(np.tile(raw_img, reps=(4, 2, 1, 1)), (0, 2, 1, 3)),
                             newshape=(4 * imgh, 2 * imgw))
        target_image = raw_img
        attn = np.transpose(attn / np.max(attn, axis=0, keepdims=True), (1, 2, 0))
        attn = cv2.resize(attn, (4 * imgh, 2 * imgw), interpolation=cv2.INTER_LANCZOS4) * 255

        self.attn = attn
        self.flag = 3

    def nextimg(self):
        if self.flag != 3:
            self.text.insert("end", "image needs to be transformed first!\n")
            return

        self.i += 1
        target_path = './display/attn/'
        img = cv2.addWeighted(self.img, 0.5, self.attn[:, :, self.i], 0.5, 0, dtype=cv2.CV_32FC1)
        cv2.imwrite(target_path + f'{self.i}.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        Img = Image.open(target_path.replace('/', '\\') + f'{self.i}.png')
        Img = Img.resize((600, 600), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(Img)
        self.imglabel = tkinter.Label(self.win, image=photo)
        self.imglabel.place(x=200, y=50)
        self.text.insert("end", "next attention\n")
        self.win.mainloop()

    def screenshot(self):
        pass


class Box:

    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

    def isNone(self):
        return self.start_x is None or self.end_x is None

    def setStart(self, x, y):
        self.start_x = x
        self.start_y = y

    def setEnd(self, x, y):
        self.end_x = x
        self.end_y = y

    def box(self):
        lt_x = min(self.start_x, self.end_x)
        lt_y = min(self.start_y, self.end_y)
        rb_x = max(self.start_x, self.end_x)
        rb_y = max(self.start_y, self.end_y)
        return lt_x, lt_y, rb_x, rb_y

    def center(self):
        center_x = (self.start_x + self.end_x) / 2
        center_y = (self.start_y + self.end_y) / 2
        return center_x, center_y


class SelectionArea:

    def __init__(self, canvas: tkinter.Canvas):
        self.canvas = canvas
        self.area_box = Box()

    def empty(self):
        return self.area_box.isNone()

    def setStartPoint(self, x, y):
        self.canvas.delete('area', 'lt_txt', 'rb_txt')
        self.area_box.setStart(x, y)
        # 开始坐标文字
        self.canvas.create_text(
            x, y - 10, text=f'({x}, {y})', fill='red', tag='lt_txt')

    def updateEndPoint(self, x, y):
        self.area_box.setEnd(x, y)
        self.canvas.delete('area', 'rb_txt')
        box_area = self.area_box.box()
        # 选择区域
        self.canvas.create_rectangle(
            *box_area, fill='black', outline='red', width=2, tags="area")
        self.canvas.create_text(
            x, y + 10, text=f'({x}, {y})', fill='red', tag='rb_txt')


class ScreenShot:

    def __init__(self, scaling_factor=2):
        self.win = tkinter.Tk()
        # self.win.tk.call('tk', 'scaling', scaling_factor)
        self.width = self.win.winfo_screenwidth()
        self.height = self.win.winfo_screenheight()

        # 无边框，没有最小化最大化关闭这几个按钮，也无法拖动这个窗体，程序的窗体在Windows系统任务栏上也消失
        self.win.overrideredirect(True)
        self.win.attributes('-alpha', 0.25)

        self.is_selecting = False

        # 绑定按 Enter 确认, Esc 退出
        self.win.bind('<KeyPress-Escape>', self.exit)
        self.win.bind('<KeyPress-Return>', self.confirmScreenShot)
        self.win.bind('<Button-1>', self.selectStart)
        self.win.bind('<ButtonRelease-1>', self.selectDone)
        self.win.bind('<Motion>', self.changeSelectionArea)

        self.canvas = tkinter.Canvas(self.win, width=self.width,
                                     height=self.height)
        self.canvas.pack()
        self.area = SelectionArea(self.canvas)
        self.win.mainloop()

    def exit(self, event):
        self.win.destroy()

    def clear(self):
        self.canvas.delete('area', 'lt_txt', 'rb_txt')
        self.win.attributes('-alpha', 0)

    def captureImage(self):
        if self.area.empty():
            return None
        else:
            box_area = [x * screen_scale_rate for x in self.area.area_box.box()]
            self.clear()
            print(f'Grab: {box_area}')
            img = ImageGrab.grab(box_area)
            return img

    def confirmScreenShot(self, event):
        img = self.captureImage()
        if img is not None:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            global target_image
            target_image = img
        self.win.destroy()

    def selectStart(self, event):
        self.is_selecting = True
        self.area.setStartPoint(event.x, event.y)
        # print('Select', event)

    def changeSelectionArea(self, event):
        if self.is_selecting:
            self.area.updateEndPoint(event.x, event.y)
            # print(event)

    def selectDone(self, event):
        # self.area.updateEndPoint(event.x, event.y)
        self.is_selecting = False


Win = window()
