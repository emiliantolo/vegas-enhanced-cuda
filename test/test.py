import os
import numpy as np
import re
import time

n_runs = 30

os.system(
    "(if not exist old mkdir old) & (move runs\\* old\\) & (if not exist runs mkdir runs)")
os.system("(if not exist build mkdir build)")

configs = [f.split(".")[0] for f in os.listdir("configs\\")]
exes = [f.split(".")[0]
        for f in os.listdir("..\\src") if re.match(r".*\.cu$", f)]

for config in configs:
    os.system("(if not exist build\\" + config +
              " mkdir build\\" + config + ")")
    os.system("(if not exist build\\" + config +
              "\\src mkdir build\\" + config + "\\src)")
    os.system("xcopy ..\\src build\\" + config + "\\src /E /Y")
    os.system("copy /Y configs\\" + config +
              ".cuh build\\" + config + "\\src\\config.cuh")
    os.system("copy /Y ..\\Makefile build\\" + config + "\\Makefile")
    os.system("cd build\\" + config + " && nmake all")

runs = np.random.permutation(n_runs * len(exes) * len(configs))

for run in runs:
    e_idx = run % len(exes)
    c_idx = run // (len(exes) * n_runs)
    exe = exes[e_idx]
    config = configs[c_idx]
    logname = config + "-" + exe + "-" + str(time.time()) + ".log"
    print("Running:", logname)
    os.system("build\\" + config + "\\bin\\" + exe +
              ".exe > runs\\" + logname)
