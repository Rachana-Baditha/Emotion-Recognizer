import tkinter as tk
from tkinter import filedialog
from tkinter.constants import BOTH, BOTTOM, CENTER, END
from PIL import ImageTk,Image
import random
import cv2
import numpy as np

root = tk.Tk()
root.title("Emotion Recognizer")
root.iconbitmap("C:\\Users\\racha\\OneDrive\\Documents\\Python\\Emotion Recognition\\VNRVJIETLogo.ico")
root.geometry("500x500")

def checkImgSize(img):

    #Getting frame dimensions
    fh = frame.winfo_height()
    fw = frame.winfo_width()

    #Getting image dimensions
    iw,ih = img.size

    #Initializing final h,w value variables
    newh = ih
    neww = iw

    if newh > fh :
        newh = fh
        scale = newh/ih
        neww = iw * scale

    if neww > fw :
        neww = fw
        scale = neww/iw
        newh = ih * scale

    return img.resize((int(neww),int(newh)), Image.ANTIALIAS)
    
def OpenImage():

    global myimg

    try:

        #User selects image to be uploaded
        imgpath = filedialog.askopenfilename(initialdir = "C:\\Users\\racha\\OneDrive\\Desktop" , title = "Select File" , 
                        filetypes = (("Images" , "*.png *.jpg *jpeg"),("all files" , "*.*")))
        
        img = Image.open(imgpath)
        img = checkImgSize(img)
        
        #Image conversion stuff cuz Tkinter is weird
        myimg = ImageTk.PhotoImage(img)

        #Removes previous image from frame
        for widget in frame.winfo_children():
            widget.destroy()
    
        #Adding image to label and packing it
        Img_label = tk.Label(frame,image = myimg)
        Img_label.pack()
    
    except AttributeError:
        return

def getEmotion():

    global result

    emote = ['Happy', 'Sad', 'Angry', 'Confused','Bored']
    result = emote[random.randint(0,4)]
    Emotion_Res.delete(0,END)
    Emotion_Res.insert(0,result)


#Creating Frame
frame = tk.Frame(root,bg = '#F5A283')
frame.place(relwidth = 0.9, relheight = 0.8, relx = 0.05, rely = 0.05)
                                    
#Creating Buttons
Open_File_Button = tk.Button(root,text = "Open Image",command = OpenImage )
Check_Emotion_Button = tk.Button(root,text = "Check Emotion", command = getEmotion)

#Adding Buttons
Open_File_Button.pack(side = BOTTOM)
Check_Emotion_Button.pack(side = BOTTOM)

#Box for Result of Emotion Checking
Emotion_Res = tk.Entry(root, width = 100)
Emotion_Res.pack(side = BOTTOM)

root.mainloop()