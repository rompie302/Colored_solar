import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def load_single_month(month):
    """Load and average single month data"""
    try:
        # Read CSV with pandas
        filename = f'Data_2016-{month:02d}.csv'
        df = pd.read_csv(filename)
        
        # Convert spectrum strings to arrays
        spectra = df['spectrum'].apply(lambda x: np.array([float(val) for val in x.split(';')]))
        spectra_array = np.stack(spectra.values)
        
        # Calculate monthly average
        return np.mean(spectra_array, axis=0)
    except Exception as e:
        print(f"Error loading month {month}: {e}")
        return None

def load_and_process_yearly_spectrum():
    """Load and process Utrecht spectrum data for all months"""
    # Load wavelengths
    wavelengths = np.loadtxt('wavelengths.csv', delimiter=';')
    
    # Process all months
    monthly_averages = []
    for month in range(1, 13):
        month_avg = load_single_month(month)
        if month_avg is not None:
            monthly_averages.append(month_avg)
    
    # Calculate yearly average
    yearly_average = np.mean(monthly_averages, axis=0)
    
    return wavelengths, yearly_average
def plot_average_spectrum(wavelengths, spectrum_avg, Font=20):
    """Plot average spectrum vs wavelength"""
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, spectrum_avg, 'b-', label='Average Spectrum')
    
    plt.xlabel('Wavelength [nm]', fontsize=Font)
    plt.ylabel('Intensity [A.U.]', fontsize=Font)
    plt.grid(True)
    plt.xticks(fontsize=Font*0.7)
    plt.yticks(fontsize=Font*0.7)
    plt.legend(fontsize=Font*0.75)
    
    plt.savefig('Average_Utrecht_Spectrum.png', dpi=400)
    plt.close()
    
def write_average_spectrum(wavelengths, spectrum_avg, filename='Average_Utrecht_spectrum.txt'):
    """Write average spectrum data to text file"""
    try:
        with open(filename, 'w') as f:
            # Write header
            f.write("Wavelength [nm] Intensity [W/(m²·nm)]\n")
            
            # Write data
            for w, s in zip(wavelengths, spectrum_avg):
                f.write(f"{w:.2f} {s:.6f}\n")
                
        print(f"Successfully wrote data to {filename}")
        
    except Exception as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":
    wavelengths, spectrum_avg = load_and_process_yearly_spectrum()
    plot_average_spectrum(wavelengths, spectrum_avg)
    write_average_spectrum(wavelengths, spectrum_avg)
    