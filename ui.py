import tkinter
import tkinter.messagebox
from tkinter.filedialog import *
from PIL import ImageTk, Image
import os
import cv2
import torch
import click
import numpy as np

import string
from pkg.utils.vocab import vocab as vocabulary
from pkg.utils.general import Config
from predict import  show_attention

from img2inchi import Img2InchiModel
from img2inchi_transformer import Img2InchiTransformerModel
from pkg.preprocess.img_process import pad_resize

class window:
    def __init__(self):
        self.win=tkinter.Tk()
        self.img=None
        self.i=0

        model_config=os.getcwd()+ '/export_config.yaml'
        config = Config(model_config)
        self.my_vocab = vocabulary(root=config.path_train_root, vocab_dir=config.vocab_dir)
        self.model=Img2InchiTransformerModel(config, output_dir='', vocab=self.my_vocab, need_output=False)
        self.model.build_pred(os.getcwd()+ '/model.ckpt', config=config)

        self.win.title('img2inchi')
        self.win.geometry('1000x1000')
        self.imgdir=None
        self.imglabel=None
        self.inchi=None
        self.text=tkinter.Text(self.win,width=85, height=7)
        self.text.place(x=200,y=650)
        B1=tkinter.Button(self.win,text="Import Image",font=('Arial',12),command=lambda:self.importimg(),width=15,height=1)
        B1.place(x=250,y=10)
        B2=tkinter.Button(self.win,text="Begin Transform",font=('Arial',12),command=lambda:self.transform(),width=15,height=1)
        B2.place(x=600,y=10)
        B3=tkinter.Button(self.win,text="next attention image",font=('Arial',12),command=lambda:self.nextimg(),width=15,height=1)
        B3.place(x=425,y=750)
        B4=tkinter.Button(self.win,text="image preprocess",font=('Arial',12),command=lambda:self.imgprocess(),width=15,height=1)
        B4.place(x=425,y=10)

        self.win.mainloop()

    def importimg(self):
        imgdir=askopenfilenames()
        if self.imglabel:
            self.imglabel.destroy()
        Img = Image.open(imgdir[0].replace('/','\\'))
        self.img=Img
        if Img.size!=(512,256):
            self.text.insert("end","The size of current image is "+str(Img.size)+"\n")
            
        self.imgdir=imgdir[0]
        photo=ImageTk.PhotoImage(Img)
        self.imglabel = tkinter.Label(self.win,image=photo)
        self.imglabel.place(x=250,y=50)
        self.text.insert("end","Import image succeed!\n")
        self.flag=0
        self.win.mainloop()

    def imgprocess(self):
        if (self.img==None):
            self.text.insert("end","No image imported!\n")
            return
        if (self.flag==2):
            self.text.insert("end","image has been processed!\n")
            return
        img = cv2.imread(self.imgdir, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        if h > w:
            img = np.rot90(img)
        pad_h, pad_v = 0, 0
        hw_ratio = (h / w) - (256 / 512)
        if hw_ratio < 0:
            pad_h = int(abs(hw_ratio) * w / 2)
        else:
            wh_ratio = (w / h) - (512 / 256)
            pad_v = int(abs(wh_ratio) * h // 2)
        img = np.pad(img, [(pad_h, pad_h), (pad_v, pad_v)], mode='constant', constant_values=255)
        img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LANCZOS4)
        img = (img / img.max() * 255).astype(np.uint8)
        img = cv2.bitwise_not(img)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        self.img=img
        target_path = './display/'
        cv2.imwrite(target_path + f'pro.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        Img = Image.open(target_path.replace('/','\\')+ f'pro.png')
        photo=ImageTk.PhotoImage(Img)
        self.imglabel = tkinter.Label(self.win,image=photo)
        self.imglabel.place(x=250,y=50)
        self.text.insert("end","process image succeed!\n")
        self.flag=1
        self.win.mainloop()

    def transform(self):
        ## TODO operate the predict function\

        if (self.flag==0):
            self.text.insert("end","image needs to be processed first!\n")
            return

        self.flag=2
        img = self.img
        img = torch.from_numpy(img).float()
        
        img = img.repeat(1, 3, 1, 1)
        result = self.model.predict(img, mode="beam")
        seq = self.my_vocab.decode(result[0])
        self.text.insert("end",seq)
        self.text.insert("end","\n")
        attn=self.model.get_attention(img, result)
        attn=attn.cpu().numpy()
        attn=attn[0,0]
        nhead, self.lenth, h, w = attn.shape
        raw_img=self.img
        assert nhead == 8
        attn = np.reshape(np.transpose(attn, (1, 0, 2, 3)), newshape=(self.lenth, 4, 2, h, w))
        attn = np.reshape(np.transpose(attn, (0, 1, 3, 2, 4)), newshape=(self.lenth, 4 * h, 2 * w))
        imgh, imgw = raw_img.shape
        imgh = int(imgh / 1.6)
        imgw = int(imgw / 1.6)
        raw_img = cv2.resize(raw_img, (imgw, imgh), interpolation=cv2.INTER_LANCZOS4)
        raw_img = np.reshape(np.transpose(np.tile(raw_img, reps=(4, 2, 1, 1)), (0, 2, 1, 3)), newshape=(4 * imgh, 2 * imgw))
        self.img=raw_img
        attn = np.transpose(attn / np.max(attn, axis=0, keepdims=True), (1, 2, 0))
        attn = cv2.resize(attn, (4 * imgh, 2 * imgw), interpolation=cv2.INTER_LANCZOS4) * 255
        
        self.attn=attn
        self.flag=3
        
        

    def nextimg(self):
        if self.flag!=3:
            self.text.insert("end","image needs to be transformed first!\n")
            return

        self.i+=1
        target_path = './display/attn/'
        img = cv2.addWeighted(self.img, 0.5, self.attn[:, :, self.i], 0.5, 0, dtype=cv2.CV_32FC1)
        cv2.imwrite(target_path + f'{self.i}.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        Img = Image.open(target_path.replace('/','\\')+ f'{self.i}.png')
        Img=Img.resize((600, 600), Image.ANTIALIAS)
        photo=ImageTk.PhotoImage(Img)
        self.imglabel = tkinter.Label(self.win,image=photo)
        self.imglabel.place(x=200,y=50)
        self.text.insert("end","next attention\n")
        self.win.mainloop()

Win=window()



