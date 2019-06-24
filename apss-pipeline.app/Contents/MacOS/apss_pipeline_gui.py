#!/usr/bin/env python

import os, sys
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter import messagebox

from pipeline import run_pipeline

do_step = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
main_data_directory = ''


if __name__ == '__main__':

    Tk().withdraw()
    main_data_directory = askdirectory()
    msg = "Processing subject: %s..." % os.path.basename(main_data_directory)
    messagebox.showinfo("Warning", msg)
    run_pipeline()
    messagebox.showinfo("Information", "Pipeline successfully completed")
    sys.exit()
