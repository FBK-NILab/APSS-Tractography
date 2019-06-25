#!/usr/bin/env python

import os, sys, time
from tkinter import Tk, Button, Checkbutton, Text
from tkinter import INSERT, BOTTOM, TOP, W, StringVar, IntVar
from tkinter.filedialog import askdirectory
from tkinter.messagebox import showinfo
from tkinter.scrolledtext import ScrolledText
from pipeline import run_pipeline

max_step = 16
do_step = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
main_data_directory = ''

step_var = []
step_label = [ \
               "1. Structural Dicom to nifti", \
               "2. Structural brain extraction", \
               "3. Diffusion DICOM to nifti", \
               "4. Diffusion brain extraction", \
               "5. Eddy current correction", \
               "6. Rescaling isotropic voxel", \
               "7. Registration of structural data", \
               "8. Registration of atlas", \
               "9. Reconstruction of tensor model", \
               "10. Deterministic tracking with DTI", \
               "11. Constraint Spherical Deconvolution", \
               "12. Deterministic tracking with CSD", \
               "13. Probabilistic tracking with CSD", \
               "14. Tractome preprocessing", \
               "15. Registration of ROI from atlas"]

win=Tk()
win.title("APSS Tractography Pipeline")
win.withdraw()
showinfo("APSS Tractography Pipeline", 'Select the folder of patient')
main_data_directory = askdirectory(title='Select the folder of patient')
win.destroy()

win=Tk()
win.title("APSS Tractography Pipeline")

def cb_sel(v):
    if v.get() == 1:
        v.set(0)
    else:
        v.set(1)

def cb_ok():
    for i in range(15):
        do_step[i+1] = step_var[i].get()
    win.destroy()

for i in range(15):
    step_var.append(IntVar())
    step_var[i].set(0)
    cb = Checkbutton(win, text=step_label[i], variable=step_var[i], command=cb_sel(step_var[i]))
    cb.pack(side=TOP, anchor=W)
cb_button = Button(win, text="OK", command=cb_ok)
cb_button.pack(side=TOP)
win.update()
win.mainloop()

win=Tk()
win.title("APSS Tractography Pipeline")
button_txt = StringVar()
button_lab = 'START'

def run():
    if button_txt.get() == 'QUIT':
        sys.exit()
    else:
        button_txt.set("QUIT")
        win.update()
        run_pipeline(main_data_directory, do_step)
        time.sleep(30)

def redirector(inputStr):
    textbox.insert(INSERT, inputStr)
    textbox.see("end")
    win.update()

sys.stdout.write = redirector 

textbox=ScrolledText(win)
textbox.pack()
subject_id = os.path.basename(main_data_directory)
textbox.insert(INSERT, "Selected patient: %s\n" % subject_id)
textbox.insert(INSERT, "Selected steps:\n")
for i in range(15):
    step_num = "   %d. " % (1 + i)
    textbox.insert(INSERT, step_num)
    if do_step[i+1]:
        step_cmd = step_label[i].split()
        textbox.insert(INSERT, " ".join(step_cmd[1:]))
    textbox.insert(INSERT, "\n")
textbox.insert(INSERT, "Press START to begin the processing of data.\n")
win.update()
button1=Button(win, textvariable=button_txt, command=run)
button_txt.set("START")
button1.pack(side=TOP)
win.update()

win.mainloop()


