#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, glob, io
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, PhotoImage, ttk, font
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import shlex, subprocess
from subprocess import Popen
from tkinter_train import *
from tkinter_test import *
from tkinter_compare import *
from tkinter_choose_create import *


# Set fonts

#ex: helv36 = tkFont.Font(family = "Helvetica",size = 36,weight = "bold")


class App(Frame):
    def __init__(self,*args,**kwargs):
       Frame.__init__(self,*args,**kwargs)
       self.notebook = ttk.Notebook()
       self.add_tab()

    def add_tab(self):
        tab1 = Train(self.notebook)
        tab2 = Compare(self.notebook)
        tab3 = Choose_Create(self.notebook)
        tab4 = Test(self.notebook) 
        self.notebook.add(tab1,text="Train", padding=5)
        self.notebook.add(tab2,text="Compare", padding=5)
        self.notebook.add(tab3,text="Choose and Create", padding=5)
        self.notebook.add(tab4,text="Test", padding=5)
        self.notebook.pack(expand=1, fill=BOTH, padx=5, pady=5)

def main(): 
    root = tk.Tk()
    root.geometry("1625x1100")
    root.title("Colourising")
    #fonts = font.Font(family="Helvetica", size=36, weight="bold")
    #font.families()
    app = App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
