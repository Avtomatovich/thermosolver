import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def roof_line():
    # AMD M1210 (1 GPU/node)
    peak_bw = 1.6 * 1e3 # GB/sec
    peak_flops = 45.3 * 1e3 # GFLOPs/sec
    peak_ai = peak_flops / peak_bw # FLOPs/byte

    ai_range = 2.0 ** np.arange(-4, 12) # [1/16, 4096]

    # giga-bytes/sec * FLOPs/byte = GFLOPs/sec, clamp with peak FLOPs
    roofline = [min(peak_flops, peak_bw * ai) for ai in ai_range]

    filenames = ['solve_data', 'diag_data']

    for filename in filenames:
        csv_filename = 'data/' + filename + '.csv'
        
        if os.path.isfile(csv_filename):
            with open(csv_filename, 'r') as csvfile:
                ai = float(next(csv.DictReader(csvfile))['arithmetic_intensity'])

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
                
                plt.title(filename.replace('_', ' ').capitalize())
                plt.savefig('plots/' + filename)
                plt.clf()

def res_plot():
    filename = 'diag_data'
    csv_filename = 'data/' + filename + '.csv'
    
    if os.path.isfile(csv_filename):
        with open(csv_filename, 'r') as csvfile:
            steps, res = [], []

            reader = csv.DictReader(csvfile)
            for row in reader:
                steps.append(int(row['steps']))
                res.append(float(row['residual']))
            
            plt.xlabel('Iteration')

            plt.yscale('log')
            plt.ylabel('Residual')
            
            # plot x=steps, y=residuals
            plt.plot(steps, res, '-r')
            
            plt.title(filename.replace('_', ' ').capitalize())
            plt.savefig('plots/' + filename)
            plt.clf()
    else:
        print(f'{csv_filename} is not a file')

if __name__ == "__main__":
    roof_line()
    res_plot()
