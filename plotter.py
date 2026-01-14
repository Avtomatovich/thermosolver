import os
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
import numpy as np

def roof_line():
    # AMD M1210 (1 GPU/node)
    peak_bw = 1.6 * 1e3 # GB/sec
    peak_flops = 45.3 * 1e3 # GFLOPs/sec
    peak_ai = peak_flops / peak_bw # FLOPs/byte

    ai_range = 2.0 ** np.arange(-4, 12) # [1/16, 4096]

    # giga-bytes/sec * FLOPs/byte = GFLOPs/sec, clamp with peak FLOPs
    roofline = [min(peak_flops, peak_bw * ai) for ai in ai_range]

    filenames = ['solve_perf', 'diag_perf']

    for filename in filenames:
        csv_filename = 'data/' + filename + '.csv'
        
        if os.path.isfile(csv_filename):
            with open(csv_filename, 'r') as csvfile:
                row = next(csv.DictReader(csvfile))
                n, ai = int(row['size']), float(row['arithmetic_intensity'])

                plt.xscale('log', base=2)
                plt.yscale('log', base=2)

                # add axis labels
                plt.xlabel('Arithmetic Intensity (FLOPs/byte)')
                plt.ylabel('Performance (GFLOPs/sec)')

                # plot x=ai_range and y=perf clamped by peak flops
                plt.plot(ai_range, roofline, 'b', label='roofline')
                # plot measured vertical AI line
                plt.axvline(ai, color="k", linestyle='--', label=f'measured AI = {ai}')
                # plot vertical AI line at roofline threshold
                plt.axvline(peak_ai, color="r", linestyle='--', label=f'threshold AI = {peak_ai}')
                plt.legend()
                
                plt.title(filename.split('_')[0].capitalize() + f' Performance at N={n}')
                plt.savefig('plots/' + filename)
                plt.clf()

def temp_plot():
    csv_filename = 'data/diag_data.csv'
    
    if os.path.isfile(csv_filename):
        with open(csv_filename, 'r') as csvfile:
            time, temp = [], []

            reader = csv.DictReader(csvfile)
            for row in reader:
                time.append(int(row['steps']))
                temp.append(float(row['total']))
            
            plt.xlabel('Time')
            plt.ylabel('Temperature')
            
            # plot x=steps, y=temp
            plt.plot(time, temp, '-r')
            
            plt.title('Heat Diffusion')
            plt.savefig('plots/temp_plot')
            plt.clf()
    else:
        print(f'{csv_filename} is not a file')

def diffuse_anim():
    filename = 'data/heat_data.dat'

    if not os.path.isfile(filename):
        print(f'{filename} is not a file')
        return

    frames = 0
    z = []

    with open(filename, 'r') as file:
        frames = int(file.readline().strip())

        rows = []
        for line in file:
            if line.strip() != '':
                rows.append([float(n) for n in line.strip().split(',')])
            else:
                z.append(rows)
                rows = []

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

    x = y = np.arange(0, len(z[0]))
    x, y = np.meshgrid(x, y)

    def frame(i):
        ax.clear()
        ax.plot_surface(x, y, np.array(z[i]), cmap=cm.inferno)
        ax.set_zlim(0, 1)
    
    anim = FuncAnimation(fig, frame, frames)

    anim.save('plots/heat_diffuse.gif', writer=PillowWriter(fps=5))

if __name__ == "__main__":
    roof_line()
    temp_plot()
    diffuse_anim()
