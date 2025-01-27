# Colored_solar
Code for coloured solar cells calculations and plots

Make sure you pip install colormath, os, scipy, numpy and matplotlib.

1) Run Final_code.py for all plots and a txt file with PCE and loss values for shifted reflection spectra. 
2) Read_utrecht_spectrum.py was used to generate the average_spectrum.txt.
3) Plot_shift.py can be used to compare losses or PCE's for different solar spectra/reflection curves.

4) If you want to use a different reflectance curve, transmittance curve or solar spectrum, feel free to do so by changing the names  in load_data in Final_code.py! You can try alot with this code probably :)
5) Do make sure you import both a reflectance and a transmittance curve of the same solar cell; there's three variables namely (transmittance, absorption, reflectance) and we only want one degree of freedom ideally! If that does not work, you can make the assumption that absorption is +- the same for difference reflectances, as we also do when we shift the spectrum.


Have fun!
