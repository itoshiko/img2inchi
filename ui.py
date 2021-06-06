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


root = None
real_resolution = get_real_resolution()
screen_size = get_screen_size()
screen_scale_rate = round(real_resolution[0] / screen_size[0], 2)
screenshot_over = False


def auto_fit(image):
    h, w = image.shape
    if h > w:
        new_w = int(float(w) / float(h) * 500)
        return cv2.resize(image, (500, new_w))
    else:
        new_h = int(float(h) / float(w) * 500)
        return cv2.resize(image, (new_h, 500))


class Window:
    def __init__(self):
        self.win = tkinter.Tk()
        self.i = 0
        global root
        root = self.win
        self.target_image = None
        self.win.title('img2inchi')
        self.win.geometry('1000x1000')
        self.win.maxsize(1000, 1000)
        self.win.minsize(1000, 1000)
        self.imglabel = None
        self.inchi = None
        self.charLabel = None
        self.text = tkinter.Text(self.win, width=85, height=7)
        self.text.place(x=200, y=700)
        self.button_open_model = tkinter.Button(self.win, text="Choose Model", font=('Arial', 12),
                                                command=lambda: self.open_model(), width=15, height=1)
        self.button_open_model.place(x=10, y=50)
        self.button_import_img = tkinter.Button(self.win, text="Import Image", font=('Arial', 12),
                                                command=lambda: self.importimg(), width=15, height=1)
        self.button_import_img.place(x=10, y=200)
        self.button_screenshot = tkinter.Button(self.win, text="Screenshot", font=('Arial', 12),
                                                command=lambda: self.screenshot(), width=15, height=1)
        self.button_screenshot.place(x=10, y=250)
        self.button_transform = tkinter.Button(self.win, text="Begin Transform", font=('Arial', 12),
                                               command=lambda: self.transform(), width=15, height=1)
        self.button_transform.place(x=10, y=450)
        self.button_transform.config(state=tkinter.DISABLED)
        self.button_next_attn = tkinter.Button(self.win, text="next attention image", font=('Arial', 12),
                                               command=lambda: self.nextimg(), width=15, height=1)
        self.button_next_attn.place(x=10, y=500)
        self.button_next_attn.config(state=tkinter.DISABLED)
        self.button_img_process = tkinter.Button(self.win, text="image preprocess", font=('Arial', 12),
                                                 command=lambda: self.imgprocess(), width=15, height=1)
        self.button_img_process.place(x=10, y=400)

        self.win.mainloop()

    def open_model(self):
        config_dir = askopenfilenames(title='Select configuration of model',
                                      filetypes=[('yaml config', '*.yaml')], initialdir='./')
        model_dir = askopenfilenames(title='Select model file',
                                     filetypes=[('model config', '*.ckpt')], initialdir='./')
        try:
            config = Config(config_dir[0])
            self.my_vocab = vocabulary(root=config.path_train_root, vocab_dir=config.vocab_dir)
            self.model = Img2InchiTransformerModel(config, output_dir='', vocab=self.my_vocab, need_output=False)
            self.model.build_pred(model_dir[0], config=config)
        except:
            self.text.insert("end", "Error occurs in opening model！")
        else:
            if self.model is not None:
                self.button_transform.config(state=tkinter.NORMAL)
                self.button_next_attn.config(state=tkinter.NORMAL)

    def importimg(self):
        img_dir = askopenfilenames()
        target_image = cv2.imread(img_dir[0].replace('/', '\\'))
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        self.target_image = target_image
        if target_image is not None:
            if self.imglabel:
                self.imglabel.destroy()
            if self.charLabel:
                self.charLabel.destroy()
            if target_image.shape != (512, 256):
                self.text.insert("end", "The size of current image is " + str(target_image.shape) + "\n")
            photo = ImageTk.PhotoImage(image=Image.fromarray(auto_fit(target_image)))
            self.imglabel = tkinter.Label(self.win, image=photo)
            self.imglabel.place(x=250, y=10)
            self.text.insert("end", "Import image succeed!\n")
            self.flag = 0
            self.win.mainloop()

    def imgprocess(self):
        target_img = self.target_image
        if target_img is None:
            self.text.insert("end", "No image imported!\n")
            return
        if self.flag == 2:
            self.text.insert("end", "image has been processed!\n")
            return
        if self.imglabel:
            self.imglabel.destroy()
        h, w = target_img.shape
        if h > w:
            target_img = np.rot90(target_img)
            h, w = target_img.shape
        pad_h, pad_v = 0, 0
        hw_ratio = (h / w) - (256 / 512)
        if hw_ratio < 0:
            pad_h = int(abs(hw_ratio) * w / 2)
        else:
            wh_ratio = (w / h) - (512 / 256)
            pad_v = int(abs(wh_ratio) * h // 2)
        target_img = np.pad(target_img, [(pad_h, pad_h), (pad_v, pad_v)], mode='constant', constant_values=255)
        target_img = cv2.resize(target_img, (512, 256), interpolation=cv2.INTER_LANCZOS4)
        target_img = (target_img / target_img.max() * 255).astype(np.uint8)
        target_img = cv2.bitwise_not(target_img)
        target_img = cv2.morphologyEx(target_img, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        self.target_image = target_img
        photo = ImageTk.PhotoImage(image=Image.fromarray(target_img))
        self.imglabel = tkinter.Label(self.win, image=photo)
        self.imglabel.place(x=250, y=10)
        self.text.insert("end", "process image succeed!\n")
        self.flag = 1
        self.win.mainloop()

    def transform(self):
        if self.flag == 0:
            self.text.insert("end", "image needs to be processed first!\n")
            return
        self.flag = 2
        target_img = self.target_image
        img = torch.from_numpy(target_img).float()
        img = img.repeat(1, 3, 1, 1)
        result = self.model.predict(img, mode="beam")
        self.raw_result = result
        seq = self.my_vocab.decode(result[0])
        self.text.insert("end", seq)
        self.text.insert("end", "\n")
        attn = self.model.get_attention(img, result)
        attn = attn.cpu().numpy()
        attn = attn[0, 0]
        nhead, self.lenth, h, w = attn.shape
        raw_img = target_img
        assert nhead == 8
        attn = np.reshape(np.transpose(attn, (1, 0, 2, 3)), newshape=(self.lenth, 4, 2, h, w))
        attn = np.reshape(np.transpose(attn, (0, 1, 3, 2, 4)), newshape=(self.lenth, 4 * h, 2 * w))
        imgh, imgw = raw_img.shape
        imgh = int(imgh / 1.6)
        imgw = int(imgw / 1.6)
        raw_img = cv2.resize(raw_img, (imgw, imgh), interpolation=cv2.INTER_LANCZOS4)
        raw_img = np.reshape(np.transpose(np.tile(raw_img, reps=(4, 2, 1, 1)), (0, 2, 1, 3)),
                             newshape=(4 * imgh, 2 * imgw))
        self.target_image = raw_img
        attn = np.transpose(attn / np.max(attn, axis=0, keepdims=True), (1, 2, 0))
        attn = cv2.resize(attn, (4 * imgh, 2 * imgw), interpolation=cv2.INTER_LANCZOS4) * 150
        attn[attn < 0] = 0
        attn[attn > 255] = 255

        self.attn = attn
        self.flag = 3

    def nextimg(self):
        if self.flag != 3:
            self.text.insert("end", "image needs to be transformed first!\n")
            return
        if self.imglabel:
            self.imglabel.destroy()
        if self.charLabel:
            self.charLabel.destroy()
        self.i += 1
        # global target_image
        target_img = self.target_image
        target_img = target_img.astype(np.uint8)
        now_attn = self.attn[:, :, self.i].astype(np.uint8)
        now_attn = cv2.applyColorMap(now_attn, cv2.COLORMAP_JET)
        now_attn[cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR) > 50] = 255
        img = cv2.cvtColor(now_attn, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((600, 600), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        self.imglabel = tkinter.Label(self.win, image=photo)
        self.imglabel.place(x=200, y=10)
        now_attn_char = ''.join(self.my_vocab.decode(self.raw_result[0, self.i].cpu().numpy().reshape(1)))
        self.charLabel = tkinter.Label(self.win,
                                       text="Char:\n" + now_attn_char[9:len(now_attn_char)], font=("Arial", 30))
        self.charLabel.place(x=830, y=500)
        self.win.mainloop()

    def screenshot(self):
        self.win.state('icon')
        global screenshot_over
        screenshot_over = False
        s = ScreenShot(self.win)
        self.win.wait_window(s.win)
        self.win.update()
        self.win.state('normal')
        global target_image
        if target_image is not None:
            if self.imglabel:
                self.imglabel.destroy()
            if self.charLabel:
                self.charLabel.destroy()
            photo = ImageTk.PhotoImage(image=Image.fromarray(auto_fit(target_image)))
            self.imglabel = tkinter.Label(self.win, image=photo)
            self.imglabel.place(x=250, y=50)
            self.text.insert("end", "Take screenshot succeed!\n")
            self.flag = 0
            self.win.mainloop()


class Box:

    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

    def is_none(self):
        return self.start_x is None or self.end_x is None

    def set_start(self, x, y):
        self.start_x = x
        self.start_y = y

    def set_end(self, x, y):
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
        return self.area_box.is_none()

    def set_start_point(self, x, y):
        self.canvas.delete('area', 'lt_txt', 'rb_txt')
        self.area_box.set_start(x, y)
        # 开始坐标文字
        self.canvas.create_text(
            x, y - 10, text=f'({x}, {y})', fill='red', tag='lt_txt')

    def update_end_point(self, x, y):
        self.area_box.set_end(x, y)
        self.canvas.delete('area', 'rb_txt')
        box_area = self.area_box.box()
        # 选择区域
        self.canvas.create_rectangle(
            *box_area, fill='black', outline='red', width=2, tags="area")
        self.canvas.create_text(
            x, y + 10, text=f'({x}, {y})', fill='red', tag='rb_txt')


class ScreenShot:
    def __init__(self, parent, scaling_factor=2):
        self.scaling_factor = scaling_factor
        self.win = tkinter.Toplevel(parent)
        # self.win.tk.call('tk', 'scaling', scaling_factor)
        self.width = self.win.winfo_screenwidth()
        self.height = self.win.winfo_screenheight()

        # 无边框，没有最小化最大化关闭这几个按钮，也无法拖动这个窗体，程序的窗体在Windows系统任务栏上也消失
        self.win.overrideredirect(True)
        self.win.attributes('-alpha', 0.25)

        self.is_selecting = False

        # 绑定按 Enter 确认, Esc 退出
        self.win.bind('<KeyPress-Escape>', self.exit)
        self.win.bind('<KeyPress-Return>', self.confirm_screenshot)
        self.win.bind('<Button-1>', self.select_start)
        self.win.bind('<ButtonRelease-1>', self.select_done)
        self.win.bind('<Motion>', self.change_selection_area)

        self.canvas = tkinter.Canvas(self.win, width=self.width,
                                     height=self.height)
        self.canvas.pack()
        self.area = SelectionArea(self.canvas)

    def exit(self, event):
        self.win.destroy()

    def clear(self):
        self.canvas.delete('area', 'lt_txt', 'rb_txt')
        self.win.attributes('-alpha', 0)

    def capture_image(self):
        if self.area.empty():
            return None
        else:
            box_area = [x * screen_scale_rate for x in self.area.area_box.box()]
            self.clear()
            print(f'Grab: {box_area}')
            img = ImageGrab.grab(box_area)
            return img

    def confirm_screenshot(self, event):
        img = self.capture_image()
        if img is not None:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            global target_image, screenshot_over
            target_image = img
        self.win.destroy()

    def select_start(self, event):
        self.is_selecting = True
        self.area.set_start_point(event.x, event.y)
        # print('Select', event)

    def change_selection_area(self, event):
        if self.is_selecting:
            self.area.update_end_point(event.x, event.y)
            # print(event)

    def select_done(self, event):
        # self.area.updateEndPoint(event.x, event.y)
        self.is_selecting = False


if __name__ == "__main__":
    win = Window()
