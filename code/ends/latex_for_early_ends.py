import os

files = os.listdir('/home/earthquakes1/homes/Rebecca/phd/stf/figures/ends/for_paper/early_end')

print(files)

for file in files:
	year = file.split('_')[0][0:4]
	month = file.split('_')[0][4:6]
	day = file.split('_')[0][6:8]
	hour = file.split('_')[1][0:2]
	minute = file.split('_')[1][2:4]
	print("\\begin{figure}")
	print("    \\centering")
	print("    \\includegraphics[width=0.8\\textwidth]{figures/stfs/ends/early_cutoff/" + file + "}")
	print("    \\caption{Normalised STF for an event " + f'{year}-{month}-{day} {hour}:{minute}' + " where the interpreted moment is much less than the total interpreted moment.}")
	print("    \\label{fig:early_end_" + file + "}")
	print("\\end{figure}")
	print("")
