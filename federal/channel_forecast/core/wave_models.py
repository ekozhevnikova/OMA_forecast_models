import pandas as pd
import numpy as np
from scipy.fft import fft, rfft, rfftfreq, ifft, fftfreq
from scipy.signal import find_peaks, argrelextrema
from sklearn.preprocessing import StandardScaler
from OMA_tools.io_data.dates import Dates_Operations
from OMA_tools.io_data.time_series import TimeSeriesTransformer
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


#################################################### ОКОННОЕ ПРЕОБРАЗОВАНИЕ ФУРЬЕ ####################################################
class TimeSeriesAnalyzer:
    def __init__(self):
        self.optimized_params = {}
    
    def _analyze_series_properties(self, series):
        """
            Анализ свойств временного ряда для автоматического подбора параметров
        """
        n = len(series)
        
        # Анализ волатильности
        volatility = np.std(series) / np.mean(series) if np.mean(series) != 0 else 1
        is_volatile = volatility > 0.3
        
        # Анализ тренда
        trend_strength = self._calculate_trend_strength(series)
        has_strong_trend = trend_strength > 0.5
        
        # Анализ сезонности
        seasonal_periods = self._find_seasonal_periods(series)
        
        # Анализ скачков (jumps)
        jump_indices = self._detect_jumps(series)
        has_jumps = len(jump_indices) > 0
        
        # Автоматический подбор window_size
        if has_strong_trend and has_jumps:
            window_size = min(60, n // 4)  # Короткое окно для volatile series
        elif len(seasonal_periods) > 0:
            # Окно, кратное основному сезонному периоду
            main_season = seasonal_periods[0]
            window_size = min(main_season * 3, n // 2)
        else:
            window_size = min(90, n // 3)
        
        # Автоматический подбор overlap
        if has_jumps or is_volatile:
            overlap = 0.8  # Большое перекрытие для лучшего отслеживания изменений
        else:
            overlap = 0.6
        
        # Автоматический подбор параметров частотного фильтра
        if has_strong_trend:
            frequency_threshold = 0.02  # Низкий порог для сохранения тренда
            num_components_ratio = 0.5  # Больше компонент
        else:
            frequency_threshold = 0.05
            num_components_ratio = 0.3
        
        return {
            'window_size': max(30, window_size),
            'overlap': overlap,
            'frequency_threshold': frequency_threshold,
            'num_components_ratio': num_components_ratio,
            'is_volatile': is_volatile,
            'has_jumps': has_jumps,
            'seasonal_periods': seasonal_periods,
            'trend_strength': trend_strength
        }
    
    def _calculate_trend_strength(self, series):
        """
            Расчет силы тренда
        """
        if len(series) < 2:
            return 0
        
        # Простой метод: отношение дисперсии тренда к общей дисперсии
        x = np.arange(len(series))
        trend = np.polyval(np.polyfit(x, series, 1), x)
        residual = series - trend
        
        var_trend = np.var(trend)
        var_total = np.var(series)
        
        return var_trend / var_total if var_total > 0 else 0
    
    def _find_seasonal_periods(self, series, max_period=365):
        """
            Поиск сезонных периодов в данных
        """
        n = len(series)
        if n < 30:
            return []
        
        # FFT для поиска периодов
        fft_vals = np.abs(fft(series - np.mean(series)))[:n // 2]
        freqs = fftfreq(n)[:n // 2]
        
        # Ищем пики в спектре
        peaks, _ = find_peaks(fft_vals, height=np.mean(fft_vals) * 1.5)
        
        seasonal_periods = []
        for peak in peaks:
            if freqs[peak] != 0:
                period = int(1 / freqs[peak])
                if 2 <= period <= min(max_period, n // 2):
                    seasonal_periods.append(period)
        
        # Сортируем по значимости
        seasonal_periods.sort()
        return seasonal_periods[:3]  # Возвращаем топ-3 периода
    
    def _detect_jumps(self, series, threshold_std=2.5):
        """
            Обнаружение скачков в данных
        """
        if len(series) < 10:
            return []
        
        # Используем разности первого порядка
        diffs = np.diff(series)
        std_diffs = np.std(diffs)
        mean_diffs = np.mean(diffs)
        
        jump_indices = []
        for i, diff in enumerate(diffs):
            if abs(diff - mean_diffs) > threshold_std * std_diffs:
                jump_indices.append(i)
        
        return jump_indices