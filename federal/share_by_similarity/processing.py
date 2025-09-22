import pandas as pd
import numpy as np
from scipy.fft import fft, rfft, rfftfreq, ifft, fftfreq
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family = 'Arial')
import warnings
warnings.filterwarnings('ignore')


class FourierForecaster:
    """
        Модель прогнозирования временных рядов на основе преобразования Фурье
    """

    def __repr__(self):
        return f'n_components: {self.n_components}'
    
    
    def __init__(self, n_components = None, threshold = 0.1):
        """
            Инициализация модели
        
            Parameters:
                n_components: int - количество гармоник для использования
                threshold: float - порог для отбора значимых частот (0-1)
        """
        self.n_components = n_components
        self.threshold = threshold
        self.coefficients_ = None
        self.frequencies_ = None
        self.signal_length_ = None
    

    @staticmethod
    def select_components_by_energy(amplitudes, energy_threshold = 0.95):
        """
            Выбор гармоник по кумулятивной энергии
        """
        # Сортируем амплитуды по убыванию
        sorted_indices = np.argsort(amplitudes)[:: -1]
        sorted_amplitudes = amplitudes[sorted_indices]
        
        # Вычисляем кумулятивную энергию
        total_energy = np.sum(sorted_amplitudes ** 2)
        cumulative_energy = np.cumsum(sorted_amplitudes ** 2) / total_energy
        
        # Находим, сколько гармоник нужно для достижения порога энергии
        n_components = np.where(cumulative_energy >= energy_threshold)[0][0] + 1
        print(f"Нужно {n_components} гармоник для {energy_threshold * 100}% энергии")
        return n_components


    def fit(self, series):
        """
            Обучение модели на временном ряде
            
            Parameters:
                series: array-like - временной ряд для обучения
        """
        series = np.array(series).flatten()
        self.signal_length_ = len(series)
        
        # Выполняем преобразование Фурье
        fft_values = rfft(series)
        frequencies = rfftfreq(self.signal_length_)
        
        # Вычисляем амплитуды
        amplitudes = np.abs(fft_values) / self.signal_length_

        self.n_components = FourierForecaster.select_components_by_energy(amplitudes)
        ## Определяем значимые частоты
        #if self.n_components is None:
        #    # Автоматический выбор компонент по порогу
        #    max_amplitude = np.max(amplitudes)
        #    significant_indices = np.where(amplitudes > self.threshold * max_amplitude)[0]
        #    
        #    # Исключаем отрицательные частоты (симметричные)
        #    positive_indices = significant_indices[significant_indices <= self.signal_length_ // 2]
        #    self.n_components = len(positive_indices)
        #    
        #    print(f"Автоматически выбрано {self.n_components} значимых гармоник")
        
        # Сохраняем коэффициенты для значимых частот
        self.coefficients_ = fft_values.copy()
        self.frequencies_ = frequencies.copy()
        
        # Обнуляем незначимые коэффициенты
        sorted_indices = np.argsort(amplitudes)[:: -1]
        keep_indices = sorted_indices[:self.n_components * 2]  # Учитываем симметрию
        
        mask = np.zeros_like(fft_values, dtype = bool)
        mask[keep_indices] = True
        self.coefficients_[~mask] = 0
        return self
    

    def predict(self, n_steps):
        """
            Прогнозирование на n_steps вперед
            
            Parameters:
                n_steps: int - количество шагов прогноза
        
            Returns:
                array - прогнозируемые значения
        """
        if self.coefficients_ is None:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        # Восстанавливаем сигнал с помощью обратного преобразования Фурье
        reconstructed = ifft(self.coefficients_)
        
        # Для прогноза экстраполируем гармоники
        time_index = np.arange(self.signal_length_ + n_steps)
        forecast = np.zeros(len(time_index), dtype = complex)
        
        # Реконструкция сигнала через сумму гармоник
        for k in range(len(self.coefficients_)):
            if np.abs(self.coefficients_[k]) > 0:  # Только значимые компоненты
                frequency = self.frequencies_[k]
                amplitude = self.coefficients_[k] / self.signal_length_
                phase = np.angle(self.coefficients_[k])
                
                # Гармоническая компонента
                component = amplitude * np.exp(1j * (2 * np.pi * frequency * time_index + phase))
                forecast += component
        return forecast.real[self.signal_length_:self.signal_length_ + n_steps]
    

    def reconstruct(self):
        """
            Реконструкция исходного ряда
        """
        if self.coefficients_ is None:
            raise ValueError("Модель не обучена.")
        
        reconstructed = ifft(self.coefficients_)
        return reconstructed.real
    

    def get_components_info(self):
        """
            Информация о значимых гармонических компонентах
        """
        amplitudes = np.abs(self.coefficients_) / self.signal_length_
        phases = np.angle(self.coefficients_)
        
        components = []
        for i in range(len(amplitudes)):
            if amplitudes[i] > 0:
                components.append({
                    'frequency': self.frequencies_[i],
                    'amplitude': amplitudes[i],
                    'phase': phases[i]
                })
        return pd.DataFrame(components).sort_values('amplitude', ascending = False)
    
    
    @staticmethod
    def plot_results(train_series, test_series, forecast, title = "Прогноз Фурье"):
        """
            Визуализация результатов прогнозирования
        """
        plt.figure(figsize = (12, 6))
        
        # Обучающая выборка
        plt.plot(range(len(train_series)), train_series, 
                label = 'Обучающие данные', color = 'blue', alpha=0.7)
        
        # Тестовая выборка
        test_start = len(train_series)
        test_end = test_start + len(test_series)
        plt.plot(range(test_start, test_end), test_series, 
                label='Реальные значения', color = 'green', alpha = 0.7)
        
        # Прогноз
        forecast_start = len(train_series)
        forecast_end = forecast_start + len(forecast)
        plt.plot(range(forecast_start, forecast_end), forecast, 
                label = 'Прогноз', color = 'red', linewidth = 2)
        
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha = 0.3)
        plt.show()
