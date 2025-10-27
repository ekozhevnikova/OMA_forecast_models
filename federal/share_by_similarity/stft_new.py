import numpy as np
import pandas as pd
from scipy.fft import fft, rfft, rfftfreq, ifft, fftfreq
from sklearn.linear_model import LinearRegression
from OMA_tools.io_data.time_series import TimeSeriesTransformer, TimeSeriesTrendAnalyze
import warnings
warnings.filterwarnings('ignore')

class FourierParametersCalculator:
    """
        Класс для расчёта параметров для оконного преобразования Фурье.
    """
    def __init__(self, series):
        self.series = series

        # Параметры, которые задаются позже
        self.window_size = None                  # размер окна
        self.overlap = None                      # процент перекрытия окон
        self.step_size = None                    # шаг перемещения окна
        self.N = len(self.series)    # длина сигнала
        self.n_components = None
    
    @staticmethod
    def round_half_up(x):
        """
            Вспомогательный метод для округления значений
        """
        return int(x + 0.5)
    

    def _calculate_data_variability(self):
        """
            Расчет изменчивости данных
        """
        # Отношение стандартного отклонения к диапазону
        data_range = np.ptp(self.series)
        if data_range > 0:
            data_variability = np.std(self.series) / data_range
            return data_variability
        else:
            data_variability = 0.5
            return data_variability


    def _set_window_size(self):
        #Для коротких ВР
        if len(self.series) < 500:
            window_size_lower = float(0.25 * len(self.series))
            window_size_upper = float(0.5 * len(self.series))
            #Среднее между верхним и нижним пределом
            self.window_size = FourierParametersCalculator.round_half_up(float(0.5 * (window_size_lower + window_size_upper)))
            print(f'Сигнал короткий! Оптимальный размер окна в промежутке {window_size_lower} - {window_size_upper}.')
            print(f'Выбираю среднее: {self.window_size}.')
            return self.window_size

        #Для средних ВР
        elif len(self.series) >= 500 and len(self.series) < 5000:
            window_size_lower = float(0.1 * len(self.series))
            window_size_upper = float(0.2 * len(self.series))
            #Среднее между верхним и нижним пределом
            self.window_size = FourierParametersCalculator.round_half_up(float(0.5 * (window_size_lower + window_size_upper)))
            print(f'Сигнал средний! Оптимальный размер окна в промежутке {window_size_lower} - {window_size_upper}.')
            print(f'Выбираю среднее: {self.window_size}.')
            return self.window_size

        #Для больших ВР
        elif len(self.series) >= 5000:
            self.window_size = FourierParametersCalculator.round_half_up(float(0.1 * len(self.series)))
            print(f'Сигнал длинный! Оптимальный размер окна {self.window_size}.')
            return self.window_size
    

    def _set_optimal_overlap(self):
        """
            Расчет оптимального перекрытия окон
        """
        data_variability = self._calculate_data_variability()
        if data_variability > 0.8:  # Высокая изменчивость
            self.overlap = 0.75
            return self.overlap  # Большее перекрытие
        elif data_variability > 0.4:  # Средняя изменчивость
            self.overlap = 0.5
            return self.overlap    # Стандартное перекрытие
        else:  # Низкая изменчивость
            self.overlap = 0.25
            return self.overlap  # Меньшее перекрытие
    

    def _set_step_size(self):
        step_size = FourierParametersCalculator.round_half_up(float(self.window_size * (1.0 - self.overlap)))
        self.step_size = max(1, step_size)  # Минимум 1 отсчет
        return self.step_size 
    

    def _calculate_n_windows(self):
        """
            Расчет количества окон
        """
        return (self.N - self.window_size) // self.step_size + 1
    

    def fourier_parameters_calculator(self):
        self.window_size = self._set_window_size()
        self.overlap = self._set_optimal_overlap()
        self.step_size = self._set_step_size()
        n_windows = self._calculate_n_windows()
        self.n_components = min(20, self.window_size // 4)
        return {
            'signal_length': self.N,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'step_size': self.step_size,
            'n_windows': n_windows,
            'n_components': self.n_components
        }


class STFT:
    """
        Класс для реализации оконного преобразования Фурье
    """
    def __init__(self, original_data, forecast_horizon, trend_window_size, sampling_rate = 1.0):
        self.original_data = original_data
        self.forecast_horizon = forecast_horizon
        self.trend_window_size = trend_window_size
        self.sampling_rate = sampling_rate

        self.series = None
        self.signal_length = None
        self.window_size = None
        self.overlap = None
        self.step_size = None
        self.n_windows  = None
        self.n_components = None

        self.trend_forecast = None
        self.detrend = False
        self.scaler = None
        self.log_transform = None


    def calculate_optimal_parameters(self):
        """
            Расчет параметров для оконного преобразования
        """
        calculator = FourierParametersCalculator(self.original_data)
        params = calculator.fourier_parameters_calculator()
        self.signal_length = params['signal_length']
        self.window_size = params['window_size']
        self.overlap = params['overlap']
        self.step_size = params['step_size']
        self.n_windows = params['n_windows']
        self.n_components = params['n_components']

        print("=== АВТОМАТИЧЕСКИ РАССЧИТАННЫЕ ПАРАМЕТРЫ ===")
        for key, value in params.items():
            print(f"{key}: {value}")
        return params


    def _validate_parameters(self):
        """
            Проверка валидности параметров
        """
        if self.window_size > self.signal_length:
            raise ValueError(f'window_size ({self.window_size}) > length данных ({self.signal_length})')
        
        if self.step_size <= 0:
            raise ValueError('step_size должен быть > 0')
        
        if self.overlap < 0 or self.overlap >= 1:
            raise ValueError('overlap должен быть в [0, 1)')
    

    def _apply_window_function(self, data, window_type = 'hann'):
        """
            Применение оконной функции
        """
        window_funcs = {
            'hann': np.hanning,
            'hamming': np.hamming,
            'blackman': np.blackman,
            'bartlett': np.bartlett,
            'none': lambda n: np.ones(n)
        }
        
        if window_type not in window_funcs:
            raise ValueError(f"Неизвестный тип окна: {window_type}")
        
        window = window_funcs[window_type](len(data))
        return data * window
    

    def _windowed_fft_analysis(self, data):
        """
            Оконное преобразование Фурье
        """
        n_windows = (len(data) - self.window_size) // self.step_size + 1
        print(n_windows)
        if n_windows <= 0:
            raise ValueError("Недостаточно данных для оконного анализа")
        
        spectra = []
        time_positions = []
        self.frequencies = fftfreq(self.window_size, 1/self.sampling_rate)

        for i in range(n_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            window_data = data[start_idx:end_idx]
            windowed_data = self._apply_window_function(window_data, 'hann')
            
            fft_result = fft(windowed_data)
            spectra.append(fft_result)
            time_positions.append(start_idx / self.sampling_rate)  # В секундах
        return np.array(spectra), np.array(time_positions)
    

    def _extract_dominant_components(self, spectra, n_components = None):
        """Извлечение доминирующих спектральных компонент"""
        if n_components is None:
            n_components = self.n_components
        
        # Усредненный спектр по времени
        mean_spectrum = np.mean(np.abs(spectra), axis = 0)
        
        # Работаем с положительными частотами
        positive_freq_mask = (self.frequencies > 0) & (self.frequencies <= self.sampling_rate / 2)
        positive_freqs = self.frequencies[positive_freq_mask]
        positive_spectrum = mean_spectrum[positive_freq_mask]
        
        # Находим наиболее значимые частоты
        dominant_indices = np.argsort(positive_spectrum)[-n_components:]
        dominant_freqs = positive_freqs[dominant_indices]
        dominant_amplitudes = positive_spectrum[dominant_indices]
        
        # Соответствующие индексы в полном спектре
        component_indices = []
        for freq in dominant_freqs:
            idx = np.argmin(np.abs(self.frequencies - freq))
            component_indices.append(idx)
        return component_indices, dominant_freqs, dominant_amplitudes
    

    def fit(self, target_column = 'Share', criteria = 0.15):
        """
            Обучение модели
        """
        #self.original_data = np.array(data).flatten()
        #self.data_length = len(self.original_data)
        
        # Проверка параметров
        #self._validate_parameters()

        # Авторасчет параметров если не установлены
        params = self.calculate_optimal_parameters()
        
        # Проверка параметров
        self._validate_parameters()
        
        print(f"\n=== ОБУЧЕНИЕ МОДЕЛИ ===")
        print(f"Данные: {self.signal_length} отсчетов")
        print(f"Окно: {self.window_size} отсчетов")
        print(f"Шаг: {self.step_size} отсчетов")
        print(f"Перекрытие: {self.overlap:.1%}")
        
        # Предобработка
        #Приведение данных к стационарному виду
        stationary_series, optimal_diff, p_value, self.log_transform = TimeSeriesTransformer(self.original_data).make_stationary()
        df_new = stationary_series.to_frame().reset_index()
        data, self.scaler = TimeSeriesTransformer.check_scale_and_modify_scale_if_need(df_new)
        ts = TimeSeriesTransformer.detect_outliers(data)
        print('-----АНАЛИЗ ТРЕНДА-----')

        self.series = self.trend_detection(ts, target_column, criteria)

        #trend_analyze = TimeSeriesTrendAnalyze(ts)
        #tau = trend_analyze.analyze_trend_comprehensive()
        #mean, ratio = trend_analyze.analyze_first_diff()
        #if (tau > 0.5 and tau <= 1) or (tau < - 0.5) or (ratio > criteria):
        #    self.detrend = True
        #    # Детрендирование
        #    detrend_series, trend = trend_analyze.extract_trend_with_ma(self.trend_window_size)

        #    # Заполняем краевые значения
        #    moving_avg = trend.ffill().bfill().values

        #    # Удаляем тренд (скользящую среднюю)
        #    detrended_series = ts[target_column] - moving_avg

        #    self.series = detrended_series
        #    # Прогноз тренда
        #    self.trend_forecast = TimeSeriesTrendAnalyze.forecast_moving_average_trend(moving_avg, self.forecast_horizon, self.trend_window_size)
        #else:
        #    self.series = stationary_series
        
        # Оконное FFT
        self.spectra, self.time_positions = self._windowed_fft_analysis(self.series)
        # Извлечение компонент
        (self.component_indices, 
         self.dominant_freqs, 
         self.dominant_amplitudes) = self._extract_dominant_components(self.spectra)
        
        print(f"Выделено {len(self.component_indices)} спектральных компонент")
        print(f"Основные частоты: {self.dominant_freqs[:5]} Гц")
        
        return self
    

    def predict(self, forecast_horizon = None):
        """
            Прогнозирование временного ряда
        """
        if forecast_horizon is None:
            forecast_horizon = self.forecast_horizon
        
        # Берем последнее окно
        last_window_start = len(self.series) - self.window_size
        last_window = self.series[last_window_start:last_window_start + self.window_size]
        
        # Эволюционный прогноз
        forecast_evolutionary = self._evolutionary_forecast_improved(last_window, self.component_indices, forecast_horizon)
        
        # Обратное преобразование
        #forecast_processed = self.scaler.inverse_transform(forecast_normalized.reshape(-1, 1)).flatten()
        
        # Восстановление тренда
        if self.detrend is not None:
            full_forecast = forecast_evolutionary + self.trend_forecast
            if self.scaler is not None and self.log_transform == True:
                X_discaled = self.scaler.inverse_transform(full_forecast.reshape(-1, 1))
                full_forecast = [i[0] for i in X_discaled]
                full_forecast = np.exp(full_forecast)
                return full_forecast
            elif self.scaler is not None and self.log_transform is not True:
                X_discaled = self.scaler.inverse_transform(full_forecast.reshape(-1, 1))
                full_forecast = [i[0] for i in X_discaled]
                return full_forecast
            elif self.scaler == None and self.log_transform == True:
                full_forecast = np.exp(full_forecast)
                return full_forecast
            
        else:
            if self.scaler is not None and self.log_transform == True:
                X_discaled = self.scaler.inverse_transform(forecast_evolutionary.reshape(-1, 1))
                forecast = [i[0] for i in X_discaled]
                forecast = np.exp(forecast)
                return forecast
            elif self.scaler is not None and self.log_transform is not True:
                X_discaled = self.scaler.inverse_transform(forecast_evolutionary.reshape(-1, 1))
                forecast = [i[0] for i in X_discaled]
                return forecast
            elif self.scaler == None and self.log_transform == True:
                forecast = [i[0] for i in X_discaled]
                return forecast

    '''
    def _evolutionary_forecast(self, last_window, component_indices, steps):
        """
            Эволюционный прогноз
        """
        current_window = last_window.copy()
        forecast = []
        
        for _ in range(steps):
            # FFT текущего окна
            windowed = self._apply_window_function(current_window)
            fft_current = fft(windowed)
            
            # Сохраняем только значимые компоненты
            fft_filtered = np.zeros_like(fft_current, dtype=complex)
            for idx in component_indices:
                fft_filtered[idx] = fft_current[idx]
                sym_idx = len(fft_current) - idx
                if sym_idx < len(fft_current):
                    fft_filtered[sym_idx] = fft_current[sym_idx]
            
            # Обратное FFT
            reconstructed = np.real(ifft(fft_filtered))
            
            # Прогнозируем следующий шаг
            next_value = reconstructed[-1]
            forecast.append(next_value)
            
            # Обновляем окно
            current_window = np.roll(current_window, -1)
            current_window[-1] = next_value
        
        return np.array(forecast)
    '''
    

    def _evolutionary_forecast_improved(self, last_window, component_indices, steps, smoothing_factor=0.1):
        """Улучшенный эволюционный прогноз с адаптацией"""
        current_window = last_window.copy()
        forecast = []
        
        # Анализ исторической точности прогноза
        historical_accuracy = self._evaluate_historical_accuracy()
        
        for step in range(steps):
            # FFT текущего окна
            windowed = self._apply_window_function(current_window)
            fft_current = fft(windowed)
            
            # Адаптивное управление компонентами
            adaptive_indices = self._adapt_components(component_indices, step, steps)
            
            # Сохраняем только значимые компоненты
            fft_filtered = np.zeros_like(fft_current, dtype=complex)
            for idx in adaptive_indices:
                fft_filtered[idx] = fft_current[idx]
                sym_idx = len(fft_current) - idx
                if sym_idx < len(fft_current):
                    fft_filtered[sym_idx] = fft_current[sym_idx]
            
            # Обратное FFT
            reconstructed = np.real(ifft(fft_filtered))
            
            # Сглаживание прогноза
            next_value = reconstructed[-1]
            if forecast and smoothing_factor > 0:
                # Учет предыдущего прогноза для сглаживания
                next_value = (1 - smoothing_factor) * next_value + smoothing_factor * forecast[-1]
            
            forecast.append(next_value)
            
            # Обновляем окно с экспоненциальным затуханием старых значений
            current_window = np.roll(current_window, -1)
            current_window[-1] = next_value
            
            # Постепенно уменьшаем влияние самых старых данных
            if step % 5 == 0 and step > 0:
                decay_factor = 0.95
                current_window[:len(current_window)//4] *= decay_factor
        
        return np.array(forecast)
    

    def _evaluate_historical_accuracy(self):
        """Оценка исторической точности прогноза"""
        # Реализуйте проверку точности на последних известных точках
        # Это поможет адаптировать параметры прогноза
        return 0.8  # Примерное значение

    def _adapt_components(self, component_indices, current_step, total_steps):
        """Адаптация компонент в процессе прогноза"""
        # Уменьшаем количество компонент по мере прогноза в будущее
        if current_step > total_steps * 0.7:
            # На поздних шагах оставляем только самые значимые компоненты
            return component_indices[:len(component_indices)//2]
        return component_indices
    

    def trend_detection(self, ts, target_column = 'Share', criteria = 0.15):
        """
            Улучшенное определение и прогноз тренда
        """
        
        # Комплексный анализ тренда
        trend_analyze = TimeSeriesTrendAnalyze(ts)
        tau = trend_analyze.analyze_trend_comprehensive()
        mean, ratio = trend_analyze.analyze_first_diff()
        
        # Более гибкие критерии для детрендинга
        should_detrend = (tau > 0.3 and tau <= 1) or (tau < -0.3) or (ratio > criteria * 0.7)
        
        if should_detrend:
            self.detrend = True
            
            # Тестируем несколько размеров окон для MA
            window_sizes = [self.trend_window_size, 14, 21, 28]
            best_window = self.trend_window_size
            best_residual_variance = float('inf')
            
            for window in window_sizes:
                try:
                    detrend_series, trend = trend_analyze.extract_trend_with_ma(window)
                    residual_variance = np.var(detrend_series)
                    
                    if residual_variance < best_residual_variance:
                        best_residual_variance = residual_variance
                        best_window = window
                except:
                    continue
            
            # Используем лучшее окно
            detrend_series, trend = trend_analyze.extract_trend_with_ma(best_window)
            moving_avg = trend.ffill().bfill().values
            detrended_series = ts[target_column] - moving_avg
            
            # Улучшенный прогноз тренда
            self.trend_forecast = self.improved_trend_forecast(moving_avg, best_window)
            self.series = detrended_series
            
            print(f"Выбрано окно тренда: {best_window}")
        else:
            self.series = ts[target_column]
            self.detrend = False
        return self.series


    def improved_trend_forecast(self, trend_series, window_size):
        """
            Улучшенный прогноз тренда
        """
        if len(trend_series) < window_size * 2:
            # Простая экстраполяция для коротких рядов
            return np.full(self.forecast_horizon, trend_series[-1])
        
        # Линейная регрессия для прогноза тренда
        X = np.arange(len(trend_series)).reshape(-1, 1)
        y = trend_series
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Прогноз с учетом последнего тренда
        last_trend_slope = np.polyfit(range(len(trend_series[-window_size:])), 
                                    trend_series[-window_size:], 1)[0]
        
        # Комбинированный прогноз
        X_future = np.arange(len(trend_series), len(trend_series) + self.forecast_horizon).reshape(-1, 1)
        linear_forecast = model.predict(X_future)
        
        # Добавляем взвешенный последний тренд
        trend_weight = min(0.5, window_size / len(trend_series))
        final_forecast = (1 - trend_weight) * linear_forecast + trend_weight * (
            trend_series[-1] + last_trend_slope * np.arange(1, self.forecast_horizon + 1)
        )
        
        return final_forecast