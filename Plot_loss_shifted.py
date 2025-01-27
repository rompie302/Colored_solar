import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
# This code is apart from the others as its better
# To write the data and then read it to make the plots
# In that way you never lose the data wooo :)
# Can compare losses of multiple spectra
def plot_combined_results(filename1, filename2, Font=20):
    """Plot color rolled values from two files with different markers"""
    # Read data from both files
    data1 = np.loadtxt(filename1, skiprows=1, delimiter=' ')
    data2 = np.loadtxt(filename2, skiprows=1, delimiter=' ')
    
    # Process first dataset
    roll1 = data1[:, 0]
    pce_black1 = data1[:, 1]
    pce_color_rolled1 = data1[:, 3]
    loss1 = 1-(pce_color_rolled1/pce_black1)
    rgb_values1 = data1[:, -3:]
    
    # Process second dataset
    roll2 = data2[:, 0]
    pce_black2 = data2[:, 1]
    pce_color_rolled2 = data2[:, 3]
    loss2 = 1-(pce_color_rolled2/pce_black2)
    rgb_values2 = data2[:, -3:]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot first dataset with circles
    for i in range(len(roll1)):
        plt.scatter(roll1[i], loss1[i]*100, 
                   c=[rgb_values1[i]], 
                   s=100,
                   marker='o',
                   label='AM 1.5' if i == 0 else "")
    
    # Plot second dataset with triangles
    for i in range(len(roll2)):
        plt.scatter(roll2[i], loss2[i]*100, 
                   c=[rgb_values2[i]], 
                   s=100,
                   marker='^',
                   label='Utrecht Spectrum' if i == 0 else "")
    
    # Format plot
    plt.xlabel('Shift [nm]', fontsize=Font)
    plt.ylabel('Loss [%]', fontsize=Font)
    plt.grid(True)
    plt.xticks(fontsize=Font*0.7)
    plt.yticks(fontsize=Font*0.7)
    plt.ylim(20,23)
    plt.legend(fontsize=Font*0.75)
    
    # Save plot
    plt.savefig('combined_results.png', dpi=400)
    plt.close()

if __name__ == "__main__":
   
    plot_combined_results('Plots/solar_integration_roll_[-100.  -80.  -60.  -40.  -20.    0.   20.   40.   60.   80.  100.].txt','Plots_Utrecht/solar_integration_roll_[-99 -79 -59 -39 -19   0  19  39  59  79  99].txt')