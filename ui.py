import tkinter
import tkinter.messagebox
from tkinter.filedialog import *
from PIL import ImageTk, Image
import string

class window:
    def __init__(self):
        self.win=tkinter.Tk()
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
        tkinter.messagebox.showinfo("Hello","hi")

Win=window()



