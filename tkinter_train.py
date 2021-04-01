#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, glob, io
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, PhotoImage, ttk, scrolledtext
from tkinter.ttk import Style
import re
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import shlex, subprocess
from subprocess import Popen


class Train(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.init_window()
#        self.progress_bar()

    ### creation of init_window ###
    def init_window(self):

        ### allowing the widget to take the full space of the root window ###
        self.pack(fill=BOTH, expand=1)
        
        
        ### buttons and widgets for EPOCH ###
        epoch_label = tk.Label(self, text="Epoch:")
        epoch_label.pack()
        epoch_label.place(x=1000, y=80)
        
        self.epoch = StringVar()
        self.epoch_widget = tk.Entry(self.master, textvariable = self.epoch, width=10)#, height=1, width=5)
        self.epoch_widget.pack()
        self.epoch_widget.place(x=1000, y=130)


        ### buttons and widgets for CONTINUE TRAINING ###
        self.continue_var = IntVar()
        continue_training = tk.Checkbutton(self.master, text="Continue training", variable=self.continue_var, command=self.check_continue_state)
        continue_training.pack()
        continue_training.place(x=60, y=170)


        ### buttons and widgets for loading IMAGE DIRECTORY ###
        image_directory_label = tk.Label(self, text="Image Directory:")
        image_directory_label.pack()
        image_directory_label.place(x=60, y=80)

        open_image_directory = tk.Button(self, text="Browse...", command=self.load_image_directory, width=10)
        open_image_directory.place(x = 770, y = 100)
        self.image_path = None
        self.image_path_widget = tk.Text(self, height=2, width=100)
        self.image_path_widget.place(x = 60, y = 100)
        self.image_path_widget.configure(state="disabled")


        ### buttons and widgets for loading WEIGHT FILE ###
        self.weight_file_label = tk.Label(self, text="Weight File:")
        self.weight_file_label.pack()
        self.weight_file_label.place(x=60, y=180)
        
        self.open_weight_file = tk.Button(self, text="Browse...", command=self.load_weight_file, width=10)
        self.open_weight_file.place(x = 770, y = 200)

        self.weight_path = None
        self.weight_path_widget = tk.Text(self, height=2, width=100)
        self.weight_path_widget.place(x = 60, y = 200)

        self.weight_file_label.configure(state="disabled")
        self.open_weight_file.configure(state="disabled")
        self.weight_path_widget.configure(state="disabled")

        
        ### buttons for START and STOP ###
        start_button = tk.Button(self, text="Train Start", command=self.start_training, width=30, height=8)
        start_button.place(x = 1350, y = 40)

        close_button = tk.Button(self, text="Stop", command=self.stop_training, width=30, height=8)
        close_button.place(x = 1350, y = 180)

    
        ### PROGRESS BAR ###
        self.s = Style(self)
# add the label to the progressbar style
        #self.s.theme_use('clam')
        self.s.layout("LabeledProgressbar",
                 [('LabeledProgressbar.trough',
                   {'children': [('LabeledProgressbar.pbar',
                                  {'side': 'left', 'sticky': 'ns'}),
                                 ("LabeledProgressbar.label",
                                  {"sticky": ""})],
                   'sticky': 'nswe'})])
        self.progress = ttk.Progressbar(self, orient="horizontal",length=1580, mode="determinate", style="LabeledProgressbar")
        self.progress.pack()
        self.progress.place(x=10, y=330)
        self.bytes = 0
        self.maxbytes = 50000
        self.progress["value"] = 0
        self.progress["maximum"] = 50000
        self.current_epoch = 0
        self.total_epoch = 0
        self.s.configure("LabeledProgressbar", text="0 %      ", foreground="white",background="red", font=("Helvetica", 15))

        progress_widget_label = tk.Label(self, text='Progress', font=("Helvetica", 15))
        progress_widget_label.pack()
        progress_widget_label.place(x=10, y=300)

        ### SCROLLED TEXT FOR STDOUT ###
        self.scrolledtext_widget = scrolledtext.ScrolledText(self, wrap="word", font=("Helvetica", 15))
        self.scrolledtext_widget.pack()
        self.scrolledtext_widget.place(x=10, y=370, height=310, width=1580)
        self.scrolledtext_widget.configure(state="disabled")
        #scrolledtext_widget_label = tk.Label(self, text=' '*25+'Original'+' '*55+'Current'+' '*55+'Target', font=("Helvetica", 15, 'bold'))
        scrolledtext_widget_label = tk.Label(self, text=' '*20+'Original'+' '*50+'Current'+' '*45+'Target', font=("Helvetica", 15, 'bold'))
        scrolledtext_widget_label.pack()
        scrolledtext_widget_label.place(x=10, y=700)
        self.current_epoch_display_label = tk.Label(self, text=' '*112+'(epoch: 0)', font=("Helvetica", 10, 'bold'))
        self.current_epoch_display_label.pack()
        self.current_epoch_display_label.place(x=10, y=720)


        self.current_img = None       
        self.current_image_frame = Label(self, image=self.current_img)
        self.current_image_frame.image = self.current_img
        self.current_image_frame.pack()
        self.current_image_frame.place(x = 5, y = 750)

        self.stop_pressed = False


        #self.current_img = Image.open('/home/june/Documents/nhk_art/sample/current.jpg')
        #self.current_img = self.current_img.resize((1584,297))
        #self.current_img = ImageTk.PhotoImage(self.current_img)
        #self.current_image_frame.configure(image=self.current_img)
        #self.current_image_frame.image=self.current_img       


    def check_continue_state(self):
        if self.continue_var.get() == 1:
            self.weight_file_label.configure(state="normal")
            self.open_weight_file.configure(state="normal")
        else:
            self.weight_file_label.configure(state="disabled")
            self.open_weight_file.configure(state="disabled")
            #self.weight_path = None
            print(self.weight_path)



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


    def load_weight_file(self):
        fname = filedialog.askopenfilename(filetypes=(("Weight files", "*.npz"),("All files", "*.*")))
        if fname:
            self.weight_path_widget.configure(state="normal")
            ### reset parameters ###        
#            if self.weight_path != None:
#                elem_pos = self.args.index('--model='+self.weight_path)
#                self.args.pop(elem_pos)            
            self.weight_path_widget.delete('1.0',END)
            
            ### insert new value ###
            self.weight_path = fname
            self.weight_path_widget.insert(END, fname)
            self.weight_path_widget.configure(state="disabled")


    def update_text(self):
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        console_line = self.proc.stdout.readline().decode('utf-8')
        console_line = ansi_escape.sub('', console_line)
#        print("[LINE ACQUIRED]:",console_line)
        
        self.scrolledtext_widget.configure(state="normal")
        self.scrolledtext_widget.insert(tk.END, console_line)
        # scrollbarを最下部に移動させる
        self.scrolledtext_widget.see(tk.END)
        self.scrolledtext_widget.configure(state="disabled")

        # バッファが空かつプロセス終了の場合は終了
        if not console_line and self.proc.poll() is not None:
            if self.stop_pressed == True:
                self.progress["value"] = 0
                self.s.configure("LabeledProgressbar", text="0 %      ")
                messagebox.showinfo('Info', 'Training stopped!')    
            else:
                messagebox.showinfo('Info', 'Training is done!')
#            return
        else: # そうでない場合は、再度呼び出し
            self.scrolledtext_widget.after(10, self.update_text)

        with open(self.image_path+'/result/log', 'r') as f:
            log_lines = []
            for log_line in f.readlines():
                log_lines.append(log_line)
            self.current_epoch = int(log_lines[-5][16:-2])
            self.update_progress()
                   
        if self.current_epoch % 10 == 0:
            self.current_img = Image.open(self.image_path + '/sample/current.jpg')
            self.current_img = self.current_img.resize((1584,297))
            self.current_img = ImageTk.PhotoImage(self.current_img)
            self.current_image_frame.configure(image=self.current_img)
            self.current_image_frame.image=self.current_img
            self.current_epoch_display_label.configure(text=' '*112+'(epoch:'+' '+str(self.current_epoch)+')', font=("Helvetica", 10, 'bold'))
        
    def start_training(self):

        ### initialise "args" ###
        args = [sys.executable, '-u','train.py','--train', '--use-original']
        self.scrolledtext_widget.configure(state="normal") 
        self.scrolledtext_widget.delete('1.0',tk.END)
        self.scrolledtext_widget.configure(state="disabled")
        self.current_img = None
        self.current_image_frame.configure(image=self.current_img)
        self.current_image_frame.image=self.current_img
        self.current_epoch_display_label.configure(text=' '*112+'(epoch: 0)', font=("Helvetica", 10, 'bold'))

        ### check if all the necessary folders exist ###
        if os.path.exists('result') == False:
            os.makedirs('result')

        if os.path.exists('sample') == False:
            os.makedirs('sample')

        if self.continue_var.get() == 1 and self.weight_path:
            args.append('--model='+self.weight_path)

        ### set the conditions for the app to run ###
        if self.continue_var.get() == 1 and self.weight_path == None:
            raise FileNotFoundError(messagebox.showerror('Input error', 'Please choose weight file'))

        ### make sure image directory is set ###
        if self.image_path:
            args.append('--data-dir='+self.image_path)
        else:
            raise FileNotFoundError(messagebox.showerror('Input error', 'Please choose image directory'))
        
#        if os.path.exists(self.image_path + '/sample/current.jpg'):
#            os.remove(self.image_path + '/sample/current.jpg')

        if os.path.exists(self.image_path + '/result/log'):
            os.remove(self.image_path + '/result/log')         


 

        ### make sure epoch is set
        if self.epoch_widget.get():
            args.append('--epoch='+self.epoch_widget.get())
        else:
            raise FileNotFoundError(messagebox.showerror('Input error', 'Please set epoch'))

        for image in glob.glob(self.image_path+'/train/'+'*.jpg'):
            if not os.path.isfile(self.image_path + '/train/orig/' + os.path.basename(image)):
                raise FileNotFoundError(messagebox.showerror('Input error', '{} not found in orig folder'.format(os.path.basename(image))))

        self.bytes = 0

        self.total_epoch = int(self.epoch_widget.get())
        
        self.proc = subprocess.Popen(args, shell=False,  stdout=subprocess.PIPE)#, stderr=subprocess.STDOUT)
        self.update_text()

        ### final check i.e. img dir clear, epoch clear and weight path clear if selected ###
        

        
    
    def update_progress(self):
        ###simulate reading 500 bytes; update progress bar###
        self.bytes = (self.current_epoch / self.total_epoch) * self.maxbytes
        self.progress["value"] = self.bytes
        self.s.configure("LabeledProgressbar", text="{0:.2f} %      ".format((self.progress["value"] / self.maxbytes) * 100))


    def stop_training(self):
        self.proc.terminate()
        self.stop_pressed = True
