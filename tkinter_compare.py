#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, glob, io
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, PhotoImage, ttk, scrolledtext
from tkinter.ttk import Style
from collections import OrderedDict
import re
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import shlex, subprocess
from subprocess import Popen
import math


class Compare(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.init_window()
#        self.progress_bar()

    ### creation of init_window ###
    def init_window(self):

        ### allowing the widget to take the full space of the root window ###
        self.pack(fill=BOTH, expand=1)     


        ### buttons and widgets for loading IMAGE DIRECTORY ###
        image_directory_label = tk.Label(self, text="Image Directory:")
        image_directory_label.pack()
        image_directory_label.place(x=20, y=10)

        open_image_directory = tk.Button(self, text="Browse...", command=self.load_image_directory, width=10)
        open_image_directory.place(x=530, y=80)

        self.image_path = None
        self.image_path_widget = tk.Text(self, height=2, width=86)
        self.image_path_widget.place(x=20, y=40)
        self.image_path_widget.configure(state="disabled")

        
        ### buttons and widgets for EPOCH ###
        epoch_label = tk.Label(self, text="Epoch:")
        epoch_label.pack()
        epoch_label.place(x=790, y=40)
        
        self.result_from = 0
        self.result_to = 0
        self.variable = StringVar(self)
        self.option = [x for x in range(self.result_from, self.result_to+1,10)]
        #self.epoch_widget = tk.OptionMenu(self, self.variable, *option)
        self.epoch_widget = ttk.Combobox(self, textvariable=self.variable, values=self.option, state="readonly")
        self.epoch_widget.pack()
        self.epoch_widget.place(x=760, y=70, width=111, height=30)

        self.weights_list = None
        
        ### buttons for START and STOP ###
#        start_button = tk.Button(self, text="Create Model", command=self.create_model, width=30, height=5)
#        start_button.place(x=440, y=550)

#        close_button = tk.Button(self, text="Stop", command=self.stop_training, width=30, height=5)
#        close_button.place(x=840, y=550)

        next_button = tk.Button(self, text=">", command=self.next_epoch, width=3, height=1)
        next_button.place(x=880, y=71)
        
        previous_button = tk.Button(self, text="<", command=self.previous_epoch, width=3, height=1)
        previous_button.place(x=700, y=71)


             
        self.current_img = None       
        self.current_image_frame = Label(self, image=self.current_img)
        self.current_image_frame.image = self.current_img
        self.current_image_frame.pack()
        self.current_image_frame.place(x = 5, y = 140)

        upper_spacer = Frame(self, height=170)
        upper_spacer.pack(side=TOP)        
        #lower_spacer = Frame(self, height=455)
        #lower_spacer.pack(side=BOTTOM)

               
        self.stop_pressed = False

        self.canvas = Canvas(self, width=1580, height=1000, scrollregion=(0, 0, 0, 0))
        self.canvas.pack(side=LEFT)
        self.canvas.place(x=10, y=140)  
        self.canvas.scrollY = Scrollbar(self, orient=VERTICAL)
        self.canvas['yscrollcommand'] = self.canvas.scrollY.set
        self.canvas.scrollY['command'] = self.canvas.yview
        self.canvas.scrollY.pack(side=RIGHT, fill=Y)


    def next_epoch(self):
        #print(self.epoch_widget.get())
        #self.variable = StringVar(self)
        #self.variable = str(int(self.epoch_widget.get()) + 10)
        current_number = list(map(str,self.option)).index(self.epoch_widget.get())
        self.epoch_widget.set(self.option[current_number+1])
        self.callback(self)
        #if int(self.epoch_widget.get()) < max(self.option):
        #    current_epoch = int(self.epoch_widget.get()) + 10
        #    self.epoch_widget.set(current_epoch)
        #    self.callback(self)
        #else:
        #    pass

    def previous_epoch(self):
#        print(self.epoch_widget.get())
        current_number = list(map(str,self.option)).index(self.epoch_widget.get())
        self.epoch_widget.set(self.option[current_number-1])
        self.callback(self)        
        #self.variable = StringVar(self)
        #self.variable = str(int(self.epoch_widget.get()) + 10)
#        if int(self.epoch_widget.get()) > min(self.option):
#            current_epoch = int(self.epoch_widget.get()) - 10
#            self.epoch_widget.set(current_epoch)
#            self.callback(self)
#        else:
#            pass


    def rounddown(self, x):
        return int(math.floor(int(x) / 100.0)) * 100

    def load_image_directory(self):
        fname = filedialog.askdirectory()
        if fname:
            self.image_path_widget.configure(state="normal")
            ### reset parameters ###
            self.image_path_widget.delete('1.0',END)

            ### insert new value ###
            self.image_path = fname
            self.image_path_widget.insert(END, fname)
            self.image_path_widget.configure(state="disabled")
            file_list = list(map(os.path.basename,sorted(glob.glob(self.image_path + "/sample/"+"*.jpg"))))
            file_list.pop(file_list.index('current.jpg'))
            
            file_list = list(map(os.path.splitext,file_list))
            file_name_list = []
            for file in file_list:
                if file[0].find("_") != -1:
                    file_name = file[0][:file[0].find("_")]
                    file_name_list.append(file_name)
                else:
                    file_name_list.append(file[0])

            filename_only_list = sorted(list(dict.fromkeys(map(int, file_name_list))))
            print('[FILE]',filename_only_list)
            #self.first_file = filename_only_list[0]
            #self.last_file = filename_only_list[-1]
  
            #self.result_from = self.first_file         
            #self.result_to = self.last_file
            self.option = filename_only_list
            #self.option = [x for x in range(self.result_from, self.result_to+1,10)]
            self.epoch_widget.configure(textvariable=self.variable, values=self.option, state="readonly")
            self.epoch_widget.bind("<<ComboboxSelected>>", self.callback)


        print("[FIRST]:",filename_only_list[0])
        print("[LAST]:",filename_only_list[-1])

    
    def read_bytes(self):
        ###simulate reading 500 bytes; update progress bar###
        self.bytes = (self.current_epoch / self.total_epoch) * self.maxbytes
#        print("[PRINT]:",self.bytes,self.current_epoch, self.total_epoch)
        #self.bytes += 500
        self.progress["value"] = self.bytes
        self.s.configure("LabeledProgressbar", text="{0:.2f} %      ".format((self.progress["value"] / self.maxbytes) * 100))
#        if self.bytes < self.maxbytes:
            # read more bytes after 100 ms
#            self.after(100, self.read_bytes)


    def stop_training(self):
        self.proc.terminate()
        self.stop_pressed = True
        


    def callback(self, eventObject):
#        print(self.epoch_widget.get())
        img_list = sorted(glob.glob(self.image_path + '/sample/'+self.epoch_widget.get()+'_'+'*.jpg'))
        imgs = [Image.open(i) for i in img_list]
        imgs_comb = np.vstack(np.asarray(i) for i in imgs)
        imgs_comb = Image.fromarray(imgs_comb)
        #self.current_img = Image.open(self.image_path + '/sample/'+self.epoch_widget.get()+'.jpg')
        #canvas = Canvas(self, width=1580, height=450, scrollregion=(0, 0, len(img_list) * 300, len(img_list) * 300))  
        

        #canvas.pack(side=LEFT)
        #canvas.place(x=15, y=170)        
        try:
            if len(img_list) > 1:
                self.canvas.configure(scrollregion=(0, 0, len(img_list) * 300, len(img_list) * 300))

    #            canvas.scrollX = Scrollbar(self, orient=HORIZONTAL)
    #            canvas['xscrollcommand'] = canvas.scrollX.set
    #            canvas.scrollX['command'] = canvas.xview
    #            canvas.scrollX.configure(side=BOTTOM, fill=X)
    #            self.canvas.scrollY.config(side=RIGHT, fill=Y)

                self.current_img = imgs_comb
                self.current_img = self.current_img.resize((1584,297*len(img_list)))
                self.current_img = ImageTk.PhotoImage(self.current_img)
                self.canvas.create_image(0, 0, image=self.current_img,anchor=NW)
            else:
                self.current_img = imgs[0]
                self.current_img = self.current_img.resize((1584,330))
                self.current_img = ImageTk.PhotoImage(self.current_img)
                self.canvas.create_image(0, 50, image=self.current_img,anchor=NW)
        except:
            #raise ValueError(print("Value error test"))
            raise ValueError(self.canvas.create_text(500,500,fill="darkblue",font="Times 20 italic bold",text="No images found"))

