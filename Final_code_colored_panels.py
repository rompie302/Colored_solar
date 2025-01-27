import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import os
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color

def integrate_spectrum(del_wavelength, spectrum):
    """Integrate spectrum"""
    # Just calculates the total spectrum
    # Assumes equal distances between data points
    # If that's not the case, use a scipy.integrate function
    return np.sum(spectrum)*del_wavelength

def integrate_visual_spectrum(wavelengths, reflected, total, x_bar, y_bar, z_bar):
    """Integrate over visual spectrum with xbar, ybar, zbar for XYZ values"""
    # Integrate total and reflected spectrum over the visual spectrum colors
    total_spectrum_x=integrate.trapezoid(y=total * x_bar, x=wavelengths)
    total_spectrum_y=integrate.trapezoid(y=total * y_bar, x=wavelengths)
    total_spectrum_z=integrate.trapezoid(y=total * z_bar, x=wavelengths)
    
    refl_x=integrate.trapezoid(y=reflected * x_bar, x=wavelengths)
    refl_y=integrate.trapezoid(y=reflected * y_bar, x=wavelengths)
    refl_z=integrate.trapezoid(y=reflected * z_bar, x=wavelengths)
    
    # Calculate XYZ based on that.
    X = refl_x/(total_spectrum_x)
    Y = refl_y/(total_spectrum_y)
    Z = refl_z/(total_spectrum_z)

    return X,Y,Z

def load_color_data(wavelengths):
    """Load and separate color matching function data"""
    # This function loads an 'objective' measure 
    # For which color one sees.
    # Once again, it is important that your data's minimum and maximum values exceed 
    # The values of the color_data.txt file for good interpolation.
    # But this should be easily doable as the color spectrum is not that large. 
    
    color_data = np.loadtxt('Color_data.txt', delimiter=' ')
    x_bar=np.interp(wavelengths, color_data[:,0], color_data[:,1])
    y_bar = np.interp(wavelengths, color_data[:,0], color_data[:,2])
    z_bar = np.interp(wavelengths, color_data[:,0], color_data[:,3])

    return {
        'x_bar': x_bar,
        'y_bar': y_bar,
        'z_bar': z_bar
    }
def process_spectrum_color(wavelengths, spectrum, total, start_visual, end_visual):
    """Process spectrum to get RGB values"""
    # Load color data
    color_data = load_color_data(wavelengths)
    
    # Extract visual range
    visual_wavelengths, visual_source,total_source, x_bar,y_bar,z_bar =  wavelengths[start_visual:end_visual], spectrum[start_visual:end_visual], total[start_visual:end_visual],color_data['x_bar'][start_visual:end_visual], color_data['y_bar'][start_visual:end_visual], color_data['z_bar'][start_visual:end_visual]
        
    # Calculate XYZ values by integrating over visual spectrum
    X, Y, Z = integrate_visual_spectrum(visual_wavelengths, visual_source,total_source,
                                      x_bar, 
                                      y_bar,
                                      z_bar)
    
    # Convert to RGB
    return convert_xyz_to_rgb(X, Y, Z)

def convert_xyz_to_rgb(X, Y, Z):
    """Convert XYZ values to RGB"""
    # Should work, if you get weird colors check it!
    xyz = XYZColor(X, Y, Z)
    rgb = convert_color(xyz, sRGBColor)
    return np.array(rgb.get_value_tuple())

def calculate_integration_results(del_wavelength, spectra):
    """Calculate integration results and losses"""
    # Calculate total currents by integrating over spectrum 
    # For black, colored, shifted color spectrum
    J_black_solar = integrate_spectrum(del_wavelength, spectra['current_black'])
    J_color_solar = integrate_spectrum(del_wavelength, spectra['current_color'])
    J_color_shifted_solar = integrate_spectrum(del_wavelength, spectra['current_color_shift'])
    # PCE is calculated by PCE=J_sc*V*FF/normal
    # IMPROVEMENT: normal is now taken to be 1000 W/m2 but this creates problems
    # For the Utrecht spectrum as the Utrecht spectrum is not normalized
    # Thus better to integrate the total spectrum from the sun (also before quantum eff etc)
    PCE_black_solar = J_black_solar*0.86*0.74/1000
    PCE_color_solar = J_color_solar*0.86*0.74/1000
    PCE_color_shifted_solar = J_color_shifted_solar*0.86*0.74/1000
    
    # Loss=(1-(J_color/J_black))*100 %                                                                 
    loss_color = (1-J_color_solar/J_black_solar)*100
    loss_shifted_color = (1-J_color_shifted_solar/J_black_solar)*100
    
    return {
        'J_black': J_black_solar,
        'J_color': J_color_solar,
        'J_color_shifted': J_color_shifted_solar,
        'PCE_black': PCE_black_solar,
        'PCE_color': PCE_color_solar,
        'PCE_color_shifted': PCE_color_shifted_solar,
        'loss_color': loss_color,
        'loss_shifted_color': loss_shifted_color
    }

def plot_spectra_comparison(Font,wavelengths, ref_fit, solar_rad, transmitted, reflected, filename):
    """Main spectral comparison plot"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    # Plot reflectance
    a = ax2.plot(wavelengths, ref_fit, label='Reflectance', color='blue')
    # Plot solar spectrum
    b = ax1.plot(wavelengths, solar_rad, label='Solar Radiation Spectrum Utrecht', color='orange')
    # Plot transmitted spectrum
    c = ax1.plot(wavelengths, transmitted, label='Transmitted Spectrum', color='green')
    # Plot reflected spectrum
    d = ax1.plot(wavelengths, reflected, label='Reflected spectrum', color='red')
    
    tot = a + b + c + d 
    labs = [l.get_label() for l in tot]
    ax1.legend(tot, labs, loc=0, fontsize=Font*0.75)
    ax1.tick_params(axis='x', labelsize=Font*0.7)
    ax1.tick_params(axis='y', labelsize=Font*0.7)
    ax1.set_xlabel('Wavelength [nm]', fontsize=Font)
    ax1.set_ylabel('Intensity [A.U.]', fontsize=Font)
    ax2.set_ylabel('Reflectance [%]', color='blue',fontsize=Font)
    ax2.set_ylim(0,1.1)
    ax1.set_ylim(0,600)
    plt.grid(True)
    plt.savefig(filename, dpi=400)
    return fig

def plot_current_comparison(Font,wavelengths, current_color, filename, current_black, rgblist):
    """Main current comparison plot"""
    plt.figure(figsize=(10, 6))
    # Plot current black solar cell and colored solar cell
    plt.plot(wavelengths,current_black,  label='black solar panel', color='black')
    plt.plot(wavelengths, current_color, label='colored solar panel', color=rgblist)
    
    plt.legend(loc=0, fontsize=Font*0.75)
    plt.xticks(fontsize=Font*0.7)
    plt.yticks(fontsize=Font*0.7)
    
    plt.xlabel('Wavelength [nm]', fontsize=Font)
    plt.ylabel(r'$J_{SC}$ [A.U.]', fontsize=Font)
    plt.grid(True)
    plt.savefig(filename, dpi=400)

def plot_transmission_comparison(Font,wavelengths, trans_orig, trans_shift, shift,image_path):
    """Plot transmission comparison"""
    plt.figure(figsize=(10, 6))
    # Plot transmission
    plt.plot(wavelengths, trans_orig, label='Transmission', color='red')
    # Plot shifted transmission
    plt.plot(wavelengths, trans_shift, label=f'Transmission shifted {int(shift)} nm', color='green')
    plt.xticks(fontsize=Font*0.7)
    plt.yticks(fontsize=Font*0.7)
    plt.xlabel('Wavelength [nm]', fontsize=Font)
    plt.ylabel('Intensity [A.U.]', fontsize=Font)
    plt.legend(fontsize=Font*0.75)
    plt.grid(True)
    plt.savefig(image_path+f"/Comparison_shifted_noshifted_{shift}.png", dpi=400)

def plot_qe_comparison(Font,qe_y, wavelengths, transmitted, current_color, image_path):
    """Plot quantum efficiency comparison"""
    # (J_photon = q lambda/hc * I) with q/hc=(1/1.2398)*10**(-3) in eV/nm
    constant=(1/1.2398)*10**(-3)
    
    # Plot quantum efficiency, current and photon current
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    # Quantum efficiency is in percentage
    a = ax2.plot(wavelengths, qe_y, label='Quantum Efficiency', color='blue')
    # Photon current is before quantum efficiency
    b = ax1.plot(wavelengths, constant*wavelengths*transmitted, label=r'$J_{photon}$', color='red')
    # Current is after quantum efficiency
    c = ax1.plot(wavelengths, current_color, label=r'$J_{electron}$', color='green')
    
    # Make legend
    tot = a + b + c 
    labs = [l.get_label() for l in tot]
    ax1.legend(tot, labs, loc=0, fontsize=Font*0.75)
    ax1.tick_params(axis='x', labelsize=Font*0.7)
    ax1.tick_params(axis='y', labelsize=Font*0.7)
    ax1.set_xlabel('Wavelength [nm]', fontsize=Font)
    ax1.set_ylabel('Current [A.U.]', fontsize=Font)
    ax2.set_ylabel('Quantum efficiency [%]', color='blue',fontsize=Font)
    ax1.set_ylim(0,270)
    ax2.set_ylim(0,1.25)
    plt.grid(True)
    plt.savefig(image_path+"/J_SC_QE_comparison.png", dpi=400)

def plot_absorption(Font,wavelengths, solar_rad, reflected, transmitted, image_path):
    """Plot absorption percentage"""
    
    # Find all non-zero values of the solar spectrum
    nonzero_mask = solar_rad != 0
    
    # Calculate absorption only for non-zero values
    absorption = np.zeros_like(wavelengths)
    absorption[nonzero_mask] = ((solar_rad[nonzero_mask] - reflected[nonzero_mask] - 
                               transmitted[nonzero_mask])/solar_rad[nonzero_mask])*100
    
    # Plot absorption
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths[nonzero_mask], absorption[nonzero_mask], label='Absorbed spectrum', color='green')
    
    plt.xticks(fontsize=Font*0.7)
    plt.yticks(fontsize=Font*0.7)
    plt.xlabel('Wavelength [nm]', fontsize=Font)
    plt.ylabel('Percentage [%]', fontsize=Font)

    plt.title('Absorbed spectrum',  fontsize=Font)
    plt.legend(fontsize=Font)    
    plt.grid(True)
    plt.savefig(image_path+"/Absorbed_spectrum.png", dpi=400)

def plot_color_visualization(rgb_values, figure_num,  image_path,shift_value):
    """Plot RGB color visualization"""
    plt.figure(figure_num)
    plt.imshow([[(rgb_values[0], rgb_values[1], rgb_values[2])]])
    plt.xticks([])
    plt.yticks([])
    rgb_str = f"_R{int(rgb_values[0]*255)}_G{int(rgb_values[1]*255)}_B{int(rgb_values[2]*255)}"
    filename = image_path+f"/Color_plot{shift_value}_{rgb_str}.png"
    plt.savefig(filename, dpi=400)

def load_data():
    """Load all input data files and return as dictionary"""
    data = {
        # File with wavelengths vs intensities
        'solar_spectrum': np.loadtxt('Average_Utrecht_Spectrum.txt', delimiter=' ',skiprows=1),
        # Files with wavelengths vs reflection, transmission and quantum efficiency 
        # Reflectance file is in percentage, so we divide by 100 in processing
        'reflection': np.loadtxt('Reflection_B4.csv', delimiter=','),
        'transmission': np.loadtxt('Transmission_B4.csv', delimiter=','),
        'quantum_eff': np.loadtxt('Quantum_eff.csv', delimiter=','),
    }
    return data

def process_data(data, amount_of_points_skipped,shift):
    """interpolate reflection, transmission and quantum efficiency data"""
    processed = {}
    
    # Wavelengths array from data
    wavelengths = data['solar_spectrum'][amount_of_points_skipped:, 0]
    processed['wavelengths'] = wavelengths
    
    # Fit polynomials
    # Probably also possible with np.interpolate but this also works
    for name, data_array in [
                            ('ref', data['reflection']), 
                           ('trans', data['transmission']),
                           ('qeff', data['quantum_eff']),
                           ]:
        x, y = data_array[:, 0], data_array[:, 1]
        
        # Reflectance file is in percentage
        if name == 'ref':
            y = y/100
        # Polynomial fit with terms up to x^12
        coeffs = np.polyfit(x, y, 12)
        poly = np.poly1d(coeffs)
        # Save values in dictionary for the wavelengths of the data
        processed[f'{name}_y'] = poly(wavelengths)
    # Shift reflection
    ref_shift = np.shift(processed['ref_y'], shift)
    # If shift is positive, fill the beginning with the first value
    if shift >= 0:
        ref_shift[:shift] = ref_shift[shift]
    # If shift is negative, fill the end with the last value
    else:
        ref_shift[shift:] = ref_shift[shift-1]
        
    processed['ref_shift_y'] = ref_shift
    
    # We assume absorption stays the same
    absorption = 1 - processed['trans_y'] - processed['ref_y']
    # Shift transmission based on reflection shift
    processed['trans_shift_y'] = 1 - ref_shift - absorption

    return processed

def calculate_spectra(amount_of_points_skipped,processed, data_solar_spectrum):
    """Calculate various spectra based on processed data"""
    spectra = {}
    
    # solar spectrum from data
    solar_spectrum=data_solar_spectrum[amount_of_points_skipped:,1]
    # Constant for current calculation, J=q lambda/hc * qeff * I  with q/hc=(1/1.2398)*10**(-3) in eV/nm
    constant=(1/1.2398)*10**(-3)
    
    # Calculate all spectra
    spectra['solar'] = solar_spectrum
    spectra['transmitted'] = processed['trans_y'] * solar_spectrum
    spectra['reflected'] = processed['ref_y'] * solar_spectrum
    spectra['reflected_shift'] = processed['ref_shift_y'] * solar_spectrum
    spectra['transmitted_shift'] = processed['trans_shift_y'] * solar_spectrum
    spectra['current_color'] = constant*processed['wavelengths']*spectra['transmitted'] * processed['qeff_y']
    spectra['current_black'] = constant*processed['wavelengths']*solar_spectrum * processed['qeff_y']
    spectra['current_color_shift'] = constant*processed['wavelengths']*spectra['transmitted_shift'] * processed['qeff_y']
    
    return spectra

def main():
    """Main execution with all plot calls"""
    
    # Constants
    # Look at the load data function if you want to change 
    # The files you're processing
    Font=20
    # (i+1) th element of array at which the visual spectrum starts and ends
    start_visual=16
    end_visual=157
    image_path = 'Plots_Utrecht'
    
    
    # If the data has zeros in the beginning, you can skip them with this parameter
    # Important: if your data's wavelength range exceeds the range of the 
    # Reflection and transmission curves, their interpolation will not work!
    # In that case, it is better to crop the data a little
    # In our case this was necessary for the low wavelengths, but it's easily adjustable 
    # For the high wavelengths.
    amount_of_points_skipped=20
    # If you want spectrum plots, set plot=1, otherwise 0
    plot=1
    # At which shift do you want to plot the spectra?
    shift_plot=100
    
    # If you want to shift the wavelength, change the values in the list below
    wavelength_shift_list=np.linspace(-100, 100, 11)
    
    # Load data, if you want different data, change the file names in the function
    data = load_data()
    
    # Calculate wavelength diff per datapoint
    del_wavelength=data['solar_spectrum'][1,0]-data['solar_spectrum'][0,0]
    
    # Determine how many datapoints we need to skip in shifting the wavelength
    shift_list=(wavelength_shift_list/del_wavelength).astype(int)
    
    # Create output file
    writefunc=(shift_list*del_wavelength).astype(int)
    output_filename = image_path+f"/solar_integration_shift_{writefunc}.txt"
    
    # Write header
    with open(output_filename, 'w') as f:
        f.write("Wavelength shift (nm), Black PCE (%), Color PCE (%), Color shifted PCE (%), Black J (A/m^2), Color J (A/m^2), Color shifted J (A/m^2), Loss color (%), Loss shifted color (%), RGB (-)\n")
        
        # Process data and calculate spectra for each shift
        for shift in shift_list:
            # Int 
            shift=int(shift)
            # Process data, interpolate:
            # - reflection
            # - transmission
            # - quantum efficiency
            # Sothat they are the same length as the Utrecht data
            processed = process_data(data, amount_of_points_skipped, shift)
            
            # Calculate reflected, absorbed, transmitted, shifted spectra
            spectra = calculate_spectra(amount_of_points_skipped,processed, data['solar_spectrum'])
            
            # Integrate the spectra and calculate PCE and losses
            integration_results = calculate_integration_results(del_wavelength, spectra)
            
            # Calculate color visualization for spectra
            rgb_list_shift = process_spectrum_color(processed['wavelengths'], spectra['reflected_shift'], spectra['solar'], start_visual,end_visual)
            
            # Plot color visualization
            plot_color_visualization(rgb_list_shift, 7,image_path, int(shift*del_wavelength))
            
            # Write results to file
            f.write(f"{shift*del_wavelength} {integration_results['PCE_black']:.5f} {integration_results['PCE_color']:.5f} {integration_results['PCE_color_shifted']:.5f} {integration_results['J_black']:.5f} {integration_results['J_color']:.5f} {integration_results['J_color_shifted']:.5f} {integration_results['loss_color']:.5f} {integration_results['loss_shifted_color']:.5f} {rgb_list_shift[0]} {rgb_list_shift[1]} {rgb_list_shift[2]}\n")
                    
        f.close()
        
    
    os.makedirs(f"{image_path}/", exist_ok=True)
    if (plot==1):
        # Process data, interpolate:
        processed = process_data(data, amount_of_points_skipped, shift_plot)
            
        # Calculate reflected, absorbed, transmitted, shifted spectra
        spectra = calculate_spectra(amount_of_points_skipped,processed, data['solar_spectrum'])

        # Calculate color visualization for spectra and shifted spectra
        rgb_list= process_spectrum_color(processed['wavelengths'], spectra['reflected'], spectra['solar'])
        rgb_list_shift = process_spectrum_color(processed['wavelengths'], spectra['reflected_shift'], spectra['solar'])
            
        # Main spectral plot
        plot_spectra_comparison(Font,processed['wavelengths'], processed['ref_y'],
                            spectra['solar'], spectra['transmitted'],
                            spectra['reflected'], 
                            image_path+'/Absorption_and_Radiation_Spectra.png')
        
        # Shifted spectral plot
        plot_spectra_comparison(Font,processed['wavelengths'], processed['ref_shift_y'],
                            spectra['solar'], spectra['transmitted_shift'],
                            spectra['reflected_shift'], 
                            image_path+f'/Absorption_and_Radiation_Spectra_shifted_{shift_plot}.png')
        
        # Current comparison plot
        plot_current_comparison(Font,processed['wavelengths'], spectra['current_color'],
                            image_path+f'/Absorption_and_Radiation_Current.png', spectra['current_black'], 
                            rgb_list)
        
        # Shifted current comparison plot
        plot_current_comparison(Font,processed['wavelengths'], spectra['current_color_shift'],
                            image_path+f'/Absorption_and_Radiation_Current_shifted_{shift}.png',spectra['current_black']
                            ,rgb_list_shift)

        # Transmission comparison shifted vs non-shifted
        plot_transmission_comparison(Font,processed['wavelengths'],
                                spectra['transmitted'],
                                spectra['transmitted_shift'],
                                shift, image_path)
        
        # QE and power conversion
        plot_qe_comparison(Font, processed['qeff_y'],
                        processed['wavelengths'],
                        spectra['transmitted'],
                        spectra['current_color'], image_path)
        
        # Absorption spectrum
        plot_absorption(Font,processed['wavelengths'],
                    spectra['solar'],
                    spectra['reflected'],
                    spectra['transmitted'], image_path)
    
if __name__ == "__main__":
    main()
