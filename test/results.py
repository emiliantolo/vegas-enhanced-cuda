import os
import numpy as np
import pandas as pd

configs = list(set([f.split("-")[0] for f in os.listdir("runs")]))
exes = list(set([f.split("-")[1] for f in os.listdir("runs")]))

results = {}

for config in configs:
    for exe in exes:
        results[(config, exe, 'tot')] = []
        results[(config, exe, 'it')] = []

for log in os.listdir("runs"):
    config = log.split("-")[0]
    exe = log.split("-")[1]
    f = open("runs\\" + log, 'r')
    for l in f:
        if l.startswith('Time'):
            results[(config, exe, 'tot')].append(float(l.split()[2]))
        if l.startswith('Iteration'):
            results[(config, exe, 'it')].append(float(l.split()[3]))

content = "Results:\n"

for config in configs:
    for exe in exes:
        tot = results[(config, exe, 'tot')]
        it = results[(config, exe, 'it')]
        cont = []
        cont.append(config + ", " + exe)
        cont.append("number: " + str(len(tot)))
        cont.append("Total:")
        cont.append("mean: " + str(np.mean(tot)))
        cont.append("std: " + str(np.var(tot) ** 0.5))
        cont.append("Iteration:")
        cont.append("mean: " + str(np.mean(it)))
        cont.append("std: " + str(np.var(it) ** 0.5))
        content += "\n\n"
        content += "\n".join(cont)

print(content)

f = open("results.log", "w")
f.write(content)
f.close()

pd.DataFrame(results).to_csv('results.csv')
