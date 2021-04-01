#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, glob, io
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, PhotoImage, ttk, scrolledtext
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import shlex, subprocess
from subprocess import Popen

class Test(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.init_window()

    ### creation of init_window ###
    def init_window(self):
        ### allowing the widget to take the full space of the root window ###
        self.pack(fill=BOTH, expand=1)
        
        ### buttons and widgets for loading IMAGE DIRECTORY ###
        image_directory_label = tk.Label(self, text="Image Directory:")
        image_directory_label.pack()
        image_directory_label.place(x=60, y=50)

        open_image_directory = tk.Button(self, text="Browse...", command=self.load_image_directory, width=10)
        open_image_directory.place(x=770, y=70)
        self.image_path = None
        self.image_path_widget = tk.Text(self, height=2, width=100)
        self.image_path_widget.place(x=60, y=70)
        self.image_path_widget.configure(state="disabled")


        ### buttons and widgets for loading WEIGHT FILE ###
        self.weight_file_label = tk.Label(self, text="Weight File:")
        self.weight_file_label.pack()
        self.weight_file_label.place(x=60, y=150)
        
        self.open_weight_file = tk.Button(self, text="Browse...", command=self.load_weight_file, width=10)
        self.open_weight_file.place(x=770, y=170)

        self.weight_path = None
        self.weight_path_widget = tk.Text(self, height=2, width=100)
        self.weight_path_widget.place(x=60, y=170)

        self.weight_file_label.configure(state="normal")
        self.open_weight_file.configure(state="normal")
        self.weight_path_widget.configure(state="disabled")

        
        ### buttons for START and STOP ###
        start_button = tk.Button(self, text="Test Start", command=self.start_test, width=24, height=5)
        start_button.place(x=1300, y=40)

        close_button = tk.Button(self, text="Stop", command=self.stop_test, width=24, height=5)
        close_button.place(x=1300, y=160)

        self.current_img = None       
        self.current_image_frame = Label(self, image=self.current_img)
        self.current_image_frame.image = self.current_img
        self.current_image_frame.pack()
        self.current_image_frame.place(x = 500, y = 650)

        ### PROGRESS BAR ###
#        self.progress = ttk.Progressbar(self, orient="horizontal",length=1280, mode="determinate")
#        self.progress.pack()
#        self.progress.place(x=10, y=330)
#        self.bytes = 0
#        self.maxbytes = 50000
#        self.progress["value"] = 0
#        self.progress["maximum"] = 50000

        ### SCROLLED TEXT FOR STDOUT ###
        self.scrolledtext_widget = scrolledtext.ScrolledText(self, wrap="word", font=("Helvetica", 15))
        self.scrolledtext_widget.pack()
        self.scrolledtext_widget.place(x=10, y=240, height=200, width=1280)
        self.scrolledtext_widget.configure(state="disabled")

        self.stop_pressed = False

        self.canvas = Canvas(self, width=1580, height=900)

        self.canvas.pack(side=LEFT)
        self.canvas.place(x=10, y=500) 


        self.scale = tk.Scale(self, orient="horizontal", command=lambda value: self.callback(value),length=1500)
        #self.scale = Scale(self, variable=var, orient=HORIZONTAL)
        #self.scale['command'] = self.callback(self)
        self.scale.pack()
        self.scale.place(x=10, y=450)
        self.scale.configure(state="disabled")



    def load_image_directory(self):
        fname = filedialog.askdirectory()
        if fname:
            self.image_path_widget.configure(state="normal")
            ### reset parameters ###
#            if self.image_path != None:
#                elem_pos = self.args.index('--data-dir='+self.image_path)
#                self.args.pop(elem_pos)
            self.image_path_widget.delete('1.0',END)

            ### insert new value ###
            self.image_path = fname
            self.image_path_widget.insert(END, fname)
            self.image_path_widget.configure(state="disabled")

            file_list = list(map(os.path.basename,sorted(glob.glob(self.image_path + "/test/"+"*.tif"))))
            file_list = list(map(os.path.splitext,file_list))
            file_name_list = []
            for file in file_list:
                file_name_list.append(file[0])
            filename_only_list = sorted(list(dict.fromkeys(map(int, file_name_list))))
            first = filename_only_list[0]
            last = filename_only_list[-1]
            self.scale.configure(from_=first, to=last, state="normal")



    def callback(self, value):
        scaler = self.scale.get()
        print(scaler)
        self.current_img = Image.open(self.image_path + '/test/'+str(value)+'.tif')
        self.current_img = self.current_img.resize((860,540))
        self.current_img = ImageTk.PhotoImage(self.current_img)
        self.canvas.create_image(300, 0, image=self.current_img,anchor=NW)


    def load_weight_file(self):
        fname = filedialog.askopenfilename(filetypes=(("Weight files", "*.npz"),("All files", "*.*")))
        if fname:
            self.weight_path_widget.configure(state="normal")
            ### reset parameters ###          
            self.weight_path_widget.delete('1.0',END)
            
            ### insert new value ###
            self.weight_path = fname
            self.weight_path_widget.insert(END, fname)
            self.weight_path_widget.configure(state="disabled")


    def update_text(self):
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        line = self.proc.stdout.readline().decode('utf-8')
        line = ansi_escape.sub('', line)
      
        self.scrolledtext_widget.configure(state="normal") 
        self.scrolledtext_widget.insert(tk.END, line)
        # scrollbarを最下部に移動させる
        self.scrolledtext_widget.configure(state="disabled")
        self.scrolledtext_widget.see(tk.END)

        # バッファが空かつプロセス終了の場合は終了
        if not line and self.proc.poll() is not None:
            if self.stop_pressed == True:
                self.progress["value"] = 0
                messagebox.showinfo('Info', 'Test stopped!')    
            else:
                messagebox.showinfo('Info', 'Test is done!')
                scaler = self.scale.get()
                self.current_img = Image.open(self.image_path + '/test/'+str(scaler)+'.tif')
                self.current_img = self.current_img.resize((860,540))
                self.current_img = ImageTk.PhotoImage(self.current_img)
                self.canvas.create_image(300, 0, image=self.current_img,anchor=NW)
            return
        else: # そうでない場合は、再度呼び出し
            self.scrolledtext_widget.after(10, self.update_text)


    def start_test(self):
        ### initialise "args" ###
        args = [sys.executable, '-u', 'train.py', '--test']
        self.scrolledtext_widget.configure(state="normal") 
        self.scrolledtext_widget.delete('1.0',tk.END)
        self.scrolledtext_widget.configure(state="disabled")

        ### check if all the necessary folders exist ###
        if os.path.exists('test') == False:
            os.makedirs('test')

        ### make sure each directory is set ###
        if self.image_path:
            args.append('--data-dir='+self.image_path)
        else:
            raise FileNotFoundError(messagebox.showerror('Input error', 'Please choose image directory'))

        if self.weight_path:
            args.append('--model='+self.weight_path)
        else:
            raise FileNotFoundError(messagebox.showerror('Input error', 'Please choose weight file'))

        self.cancel_id = None

        #print('[ATTENTION]: Pressed!!!')
#        self.update_text()

        self.proc = subprocess.Popen(args, shell=False,  stdout=subprocess.PIPE)#, stderr=subprocess.STDOUT)
        self.update_text()


    def stop_test(self):
        self.proc.terminate()
        self.stop_pressed = True
