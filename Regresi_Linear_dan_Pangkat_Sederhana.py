import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

print("||      PROGRAM IMPLEMENTASI REGRESI     ||")
print("||      LINEAR DAN PANGKAT SEDERHANA     ||")
print("||   ALMAN KAMAL MAHDI - 21120122120024  ||")
print("||         METODE NUMERIK KELAS B        ||")

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.rms_error = None
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def calculate_rms_error(self, y_true, y_pred):
        self.rms_error = np.sqrt(mean_squared_error(y_true, y_pred))
        return self.rms_error

class PowerLawModel:
    def __init__(self):
        self.params = None
        self.rms_error = None
    
    def fit(self, X, y):
        def power_law(x, a, b):
            return a * np.power(x, b)
        self.params, _ = curve_fit(power_law, X.flatten(), y.values, p0=[1, 1])
    
    def predict(self, X):
        a, b = self.params
        return a * np.power(X.flatten(), b)
    
    def calculate_rms_error(self, y_true, y_pred):
        self.rms_error = np.sqrt(mean_squared_error(y_true, y_pred))
        return self.rms_error

class RegressionComparison:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X = self.data[['Hours Studied']].values
        self.y = self.data['Performance Index']
        self.linear_model = LinearRegressionModel()
        self.power_model = PowerLawModel()
    
    def fit_models(self):
        self.linear_model.fit(self.X, self.y)
        self.power_model.fit(self.X, self.y)
    
    def predict_and_evaluate(self):
        # Model Linear
        y_pred_linear = self.linear_model.predict(self.X)
        rms_linear = self.linear_model.calculate_rms_error(self.y, y_pred_linear)
        
        # Model Pangkat
        y_pred_power = self.power_model.predict(self.X)
        rms_power = self.power_model.calculate_rms_error(self.y, y_pred_power)
        
        return rms_linear, rms_power
    
    def plot_results(self, rms_linear, rms_power):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Plot regresi linear
        x_range_linear = np.linspace(1, 9, 100).reshape(-1, 1)
        y_predict_linear_range = self.linear_model.predict(x_range_linear)
        axs[0].scatter(self.X, self.y, color='black', label='Data', s=0.8)
        axs[0].plot(x_range_linear, y_predict_linear_range, color='cyan', label='Regresi Linear')
        axs[0].set_xlabel('Jam Belajar')
        axs[0].set_ylabel('Indeks Performa')
        axs[0].set_title(f'RMS Error Linear: {rms_linear:.5f}')
        axs[0].legend()
        axs[0].xaxis.set_ticks(np.arange(1, 10, 1))
        axs[0].yaxis.set_ticks(np.arange(self.y.min(), self.y.max() + 1, 5))

        # Plot regresi pangkat
        x_range_power = np.linspace(1, 9, 100)
        y_predict_power_range = self.power_model.predict(x_range_power)
        axs[1].scatter(self.X, self.y, color='black', label='Data', s=0.8)
        axs[1].plot(x_range_power, y_predict_power_range, color='red', label='Regresi Pangkat')
        axs[1].set_xlabel('Jam Belajar')
        axs[1].set_ylabel('Indeks Performa')
        axs[1].set_title(f'RMS Error Pangkat: {rms_power:.5f}')
        axs[1].legend()
        axs[1].xaxis.set_ticks(np.arange(1, 10, 1))
        axs[1].yaxis.set_ticks(np.arange(self.y.min(), self.y.max() + 1, 5))

        # Tambahkan judul utama untuk seluruh gambar
        fig.suptitle('Hasil Regresi Linear dan Pangkat', fontsize=16)

        # Tampilkan plot di tengah layar
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def main():
    # Mengambil data dari file .csv
    data_path = 'Student_Performance.csv'
    comparison = RegressionComparison(data_path)
    comparison.fit_models()
    rms_linear, rms_power = comparison.predict_and_evaluate()
    comparison.plot_results(rms_linear, rms_power)

    print(f'RMS Error Linear: {rms_linear:.5f}')
    print(f'RMS Error Power Law: {rms_power:.5f}')

    if rms_linear < rms_power:
        print('Regresi Linear memberikan error RMS yang lebih rendah.')
    else:
        print('Regresi Pangkat memberikan error RMS yang lebih tinggi.')

if __name__ == "__main__":
    main()
