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
from predict import  pre_process,interactive_shell

from img2inchi import Img2InchiModel
from img2inchi_transformer import Img2InchiTransformerModel
from pkg.preprocess.img_process import pad_resize

class window:
    def __init__(self):
        self.win=tkinter.Tk()
        self.img=None

        model_config=os.getcwd()+ '/export_config.yaml'
        config = Config(model_config)
        self.my_vocab = vocabulary(root=config.path_train_root, vocab_dir=config.vocab_dir)
        self.model=Img2InchiTransformerModel(config, output_dir='', vocab=self.my_vocab, need_output=False)
        self.model.build_pred(os.getcwd()+ '/model.ckpt', config=config)

        self.win.title('img2inchi')
        self.win.geometry('800x600')
        self.imgdir=None
        self.imglabel=None
        self.inchi=None
        self.text=tkinter.Text(self.win,width=85, height=8)
        self.text.place(x=100,y=450)
        B1=tkinter.Button(self.win,text="Import Image",font=('Arial',12),command=lambda:self.importimg(),width=15,height=1)
        B1.place(x=250,y=10)
        B2=tkinter.Button(self.win,text="Begin Transform",font=('Arial',12),command=lambda:self.transform(),width=15,height=1)
        B2.place(x=400,y=10)

        self.win.mainloop()
    def importimg(self):
        imgdir=askopenfilenames()
        if self.imglabel:
            self.imglabel.destroy()
        Img = Image.open(imgdir[0].replace('/','\\'))
        if Img.size!=(512,256):
            self.text.insert("end","Wrong image size! The size of current image is "+str(Img.size)+"\n")
        else:
            self.imgdir=imgdir[0]
            photo=ImageTk.PhotoImage(Img)
            self.imglabel = tkinter.Label(self.win,image=photo)
            self.imglabel.place(x=144,y=50)
            self.text.insert("end","Import image succeed!\n")
            self.win.mainloop()
    

    def transform(self):
        ## TODO operate the predict function
        self.image = cv2.imread(self.imgdir, cv2.IMREAD_GRAYSCALE)
        _config = Config('./config/data_prepare.yaml')
        image = (self.image / self.image.max() * 255).astype(np.uint8)
        image = cv2.threshold(image, _config.threshold, 255, cv2.THRESH_BINARY)
        img = torch.from_numpy(self.image).float()
        img = img.repeat(1, 3, 1, 1)
        result = self.model.predict(img, mode="beam")
        seq = self.my_vocab.decode(result[0])
        self.text.insert("end",seq)

Win=window()



