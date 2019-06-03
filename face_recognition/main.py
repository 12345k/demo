import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import face_detection.vggface as face
import os
from helper import FPS2


# face.traing()


width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()

lmain1 = tk.Label(root)
lmain1.pack()


normal_button = tk.Button(master=root, text='Back',command=lambda:[normal_fun(),erase_all_button(),show_normal_buttons()])
normal_button.pack()

people_button = tk.Button(master=root, text='people detection',command=lambda:[people_fun(),erase_all_button(),show_people_buttons()])
people_button.pack()

face_button = tk.Button(master=root, text='face recognition',command=lambda:[face_fun(),erase_all_button(),show_face_buttons()])
face_button.pack()



name = tk.Label(master=root,text='Name')
name.pack()

name_field = tk.Entry(master=root)
name_field.pack()


register_button = tk.Button(master=root, text='Register',command=lambda:register_fun())
register_button.pack()

back_button = tk.Button(master=root, text='Back',command=lambda:back_fun())
back_button.pack()

quit_button = tk.Button(master=root, text='Quit',command=lambda: quit_(root))
quit_button.pack()

register_button.pack_forget()
back_button.pack_forget()
name_field.pack_forget()
name.pack_forget()
normal_button.pack_forget()

flag = 0
name_string = ""



def back_fun():
  quit_button.pack_forget()
  register_button.pack_forget()
  back_button.pack_forget()
  name.pack_forget()
  name_field.pack_forget()

  normal_button.pack()
  people_button.pack()
  face_button.pack()
  quit_button.pack()



  global flag
  flag=0
  return True




def register_fun():

  global flag,name_string
  name_string = str(name_field.get())
  name_field.delete(0,tk.END)
  flag=3
  return True

def normal_fun():
  global flag
  flag = 0
  return flag


def people_fun():
  global flag
  flag = 1
  return flag

def face_fun():
  global flag
  flag = 2
  return flag

def quit_(root):
    root.destroy()


def erase_all_button():
  register_button.pack_forget()
  back_button.pack_forget()
  name.pack_forget()
  name_field.pack_forget()
  normal_button.pack_forget()
  people_button.pack_forget()
  face_button.pack_forget()
  quit_button.pack_forget()

def show_people_buttons():

  face_button.pack()
  normal_button.pack()
  quit_button.pack()

def show_normal_buttons():
   people_button.pack()
   face_button.pack()
   quit_button.pack()

def show_face_buttons():

  name.pack()
  name_field.pack()
  register_button.pack()
  back_button.pack()
  quit_button.pack()


def show_frame():

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    global flag
    if flag == 1:
      frame = people_detection(frame)
    elif flag == 2:
      frame = face_detection(frame)
    elif flag == 3:
      face_cascade = cv2.CascadeClassifier('E:/code/face_recognition/face_detection/haarcascade_frontalface_default.xml')
      faces = face_cascade.detectMultiScale(frame, 1.3, 5)
      for (x,y,w,h) in faces:
        detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
        cv2.imwrite("E:/code/face_recognition/face_detection/images/"+name_string+".jpg",detected_face)
      face.training()
      flag = 2
    elif flag ==0:
      normal_button.pack_forget()


    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # fps_text = "FPS: "+str(fps)
    cv2.putText(cv2image,"FPS: {}".format(str(fps.fps_local())), (15,30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
    fps.update()
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain1.imgtk = imgtk
    lmain1.configure(image=imgtk)
    lmain1.after(10, show_frame)


def people_detection(frame):
  import object_detection.deep_learning_object_detection as obj
  frame = obj.main(frame)
  normal_button.pack()
  return frame

def face_detection(frame):

  frame  = face.read_image(frame)
  return frame






fps = FPS2(5).start()
show_frame()

root.mainloop()
