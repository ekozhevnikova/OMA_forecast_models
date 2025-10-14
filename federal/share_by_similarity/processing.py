import pandas as pd
import numpy as np
from scipy.fft import fft, rfft, rfftfreq, ifft, fftfreq
from sklearn.preprocessing import StandardScaler
from OMA_tools.io_data.dates import Dates_Operations
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



class STFFT(FourierForecaster):
    """
        Оконное преобразование Фурье - дочерний класс от FourierForecaster
        Сохраняет все методы родителя + добавляет оконный анализ
    """
    def __init__(self, series,
                 n_components = None, threshold = 0.1, 
                 window_type = 'hann', use_adaptive_components = True, use_trend = True, trend_window = None):
        """
        Инициализация улучшенного оконного прогнозирутеля
        
        Parameters:
            series: pd.DataFrame, где столбец с индексом - столбец с Датой.
            n_components, threshold: наследуются от родителя
            window_type: тип оконной функции
            use_adaptive_components: адаптивный выбор компонент для каждого окна
        """
        # Инициализация родительского класса
        super().__init__(n_components, threshold)
        
        self.series = series
        self.window_type = window_type
        self.use_adaptive_components = use_adaptive_components
        self.use_trend = use_trend
        self.trend_window = trend_window or self._auto_trend_window()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Атрибуты для оконного анализа
        # размер окна для анализа
        self.window_size = None
        # шаг окна
        self.step_size = None
        # процент перекрытия окон (0-1)
        self.overlap = None
        # частота дискретизации - количество отсчетрв сигнала в секунду
        self.sampling_rate = None
        # Длина сигнала
        self.signal_length_ = None

        self.window_coefficients_ = None
        self.window_frequencies_ = None
        self.window_positions_ = None
        self.dominant_frequencies_ = None
        self.dominant_amplitudes_ = None
    

    def _normalize_data(self):
        """
            Нормализация данных
        """
        data_2d = self.series.reshape(-1, 1)
        normalized = self.scaler.fit_transform(data_2d).flatten()
        self.is_fitted = True
        return normalized


    @staticmethod
    def build_window_size(series):
        """
            Функция для выбора размера окна, основываясь на длине сигнала.
            Args:
            Returns:
        """
        #Вспомогательная функция для округления
        def round_half_up(x):
                return int(x + 0.5)
            
        #Для коротких ВР
        if len(series) < 500:
            window_size_lower = round_half_up(float(0.25 * len(series)))
            window_size_upper = round_half_up(float(0.5 * len(series)))
            #Среднее между верхним и нижним пределом
            window_size = 0.5 * (window_size_lower + window_size_upper)
            print(f'Сигнал короткий! Оптимальный размер окна в промежутке {window_size_lower} - {window_size_upper}.')
            print(f'Выбираю среднее: {window_size}.')
            return int(window_size)
        #Для средних ВР
        elif len(series) >= 500 and len(series) < 5000:
            window_size_lower = round_half_up(float(0.1 * len(series)))
            window_size_upper = round_half_up(float(0.2 * len(series)))
            #Среднее между верхним и нижним пределом
            window_size = 0.5 * (window_size_lower + window_size_upper)
            print(f'Сигнал средний! Оптимальный размер окна в промежутке {window_size_lower} - {window_size_upper}.')
            print(f'Выбираю среднее: {window_size}.')
            return int(window_size)
        #Для больших ВР
        elif len(series) >= 5000:
            window_size = round_half_up(float(0.1 * len(series)))
            print(f'Сигнал длинный! Оптимальный размер окна {window_size}.')
            return int(window_size)


    @staticmethod
    def overlap_selections(window_size):
        """
            Функция для генерации процента перекрытия окон
        """
        #Маленькие окна (С окном < 64 нельзя различить частоты ближе, чем ~15 Гц при типичных sampling_rate)
        if window_size < 64:
            return 0.75
        #Большие окна
        elif window_size > 512:
            return 0.6
        #Средние окна
        else:
            return 0.75
        

    @staticmethod
    def calculate_step_size(window_size, overlap):
        """
            Функция для расчёта шага перемещения окна
        """
        #Если overlap указан в процентах (например, 75)
        if overlap > 1.0:
            overlap_ = overlap / 100
            step_size = int(window_size * (1.0 - overlap_))
            return int(step_size)
        #Если overlap указан в долях (например, 0.75)
        else:
            step_size = int(window_size * (1.0 - overlap))
            return int(step_size)


    @staticmethod
    def calculate_sampling_rate(total_samples, date_start, date_stop):
        """
            Функция для расчета частоты дискретизации (sampling_rate), исходя из общего количества измерений и длительности.
            Для измерений раз в день sampling_rate может быть около нуля.
            Args:
                total_samples: общее количество измерений
                date_start: дата старта в формате Timestamp
                date_stop: дата конца в формате Timestamp
                durations_seconds: общая длительность измерения в секундах
            Returns:
                sampling_rate: частота дискретизации в Гц
        """
        # Для дневных данных используем дни как основную единицу
        total_days = (date_stop - date_start).days
        if total_days > 0:
            sampling_rate = total_samples / total_days  # измерений в день
        else:
            sampling_rate = 1.0  # fallback
        # Если sampling_rate слишком мал, используем нормализованные частоты
        if sampling_rate < 0.001:
            print(f"Внимание: низкий sampling_rate ({sampling_rate:.6f}), используются нормализованные частоты")
        
        return sampling_rate


    def _select_components_by_energy(amplitudes, energy_threshold = 0.95):
        """
        Выбор компонент основанный на доле энергии которую они объясняют.
        
        Parameters:
            amplitudes: амплитуды гармоник (отсортированные)
            energy_threshold: порог кумулятивной энергии (0 - 1)
        Returns:
            n_components: количество значимых компонент
        """
        # Сортируем амплитуды по убыванию
        sorted_indices = np.argsort(amplitudes)[::-1]
        sorted_amplitudes = amplitudes[sorted_indices]
        
        # Вычисляем энергию (квадраты амплитуд)
        energies = sorted_amplitudes ** 2
        total_energy = np.sum(energies)
        
        # Кумулятивная энергия
        cumulative_energy = np.cumsum(energies) / total_energy
        
        # Находим где достигается порог
        n_components = np.argmax(cumulative_energy >= energy_threshold) + 1    
        return int(n_components)


    def _create_window_function(self):
        """
            Создание оконной функции
        """
        if self.window_type == 'hann':
            return np.hanning(self.window_size)
        elif self.window_type == 'hamming':
            return np.hamming(self.window_size)
        elif self.window_type == 'blackman':
            return np.blackman(self.window_size)
        elif self.window_type == 'rectangular':
            return np.ones(self.window_size)
        else:
            print(f"Неизвестное окно '{self.window_type}', используется Ханн")
            return np.hanning(self.window_size)
    

    def _windowed_fit(self, date_start, date_stop):
        """
            Оконное обучение с улучшенной предобработкой
        """
        print("=== ОКОННОЕ ОБУЧЕНИЕ ===")

        # 2. Оконное преобразование Фурье
        self.perform_windowed_analysis(date_start, date_stop)
        
        # 3. Анализ результатов
        #self._analyze_window_results()
        
        print(f"Обучение завершено: {len(self.window_positions_)} окон")
        return self


    def fit(self, date_start, date_stop):
        """
        Улучшенный метод fit с поддержкой обоих режимов
        
        Parameters:
            target_column: название столбца с таргетом
        """
        self.series = np.array(self.series).flatten()
        return self._windowed_fit(date_start, date_stop)


    def perform_windowed_analysis(self, date_start, date_stop):
        """
            Выполнение оконного преобразования Фурье.
            Args:
                target_column: название столбца с таргетом.
                date_start: дата старта в формате Timestamp, нужно для расчета sampling_rate
                date_stop: дата конца в формате Timestamp, нужно для расчета sampling_rate
        """
        n = len(self.series)
        self.signal_length_ = len(self.series)

        # 1. НОРМАЛИЗАЦИЯ ДАННЫХ - ДОБАВЛЯЕМ ЭТОТ ШАГ
        normalized_series = self._normalize_data()
        #НОВОЕ
        # 1. Выделение тренда
        if self.use_trend:
            self.trend_component_, detrended_series = self._extract_trend(normalized_series)
            self.detrended_series_ = detrended_series
            analysis_series = detrended_series
            print(f"Тренд выделен (окно {self.trend_window}), анализируется детрендированный ряд")
        else:
            analysis_series = normalized_series
            print("Анализ без выделения тренда")

        # 2. Параметры оконного анализа (используем детрендированный ряд)
        self.window_size = STFFT.build_window_size(analysis_series)
        self.overlap = STFFT.overlap_selections(self.window_size)
        self.step_size = STFFT.calculate_step_size(self.window_size, self.overlap)
        self.sampling_rate = STFFT.calculate_sampling_rate(n, date_start, date_stop)

        # 3. Оконный анализ (остальной код без изменений, но используем analysis_series)
        n_windows = int((n - self.window_size) // self.step_size + 1)
        
        self.window_coefficients_ = []
        self.window_frequencies_ = []
        self.window_positions_ = []
        self.dominant_frequencies_ = []
        self.dominant_amplitudes_ = []
        
        window_func = self._create_window_function()
        
        print(f"Анализ {n_windows} окон...")
        
        for i in range(n_windows):
            start = int(i * self.step_size)
            end = int(start + self.window_size)
            position = start + self.window_size // 2
            
            # Используем детрендированный ряд для анализа
            window_data = analysis_series[start:end] * window_func
            # FFT для окна
            coeffs = rfft(window_data)
            freqs = rfftfreq(self.window_size, 1 / self.sampling_rate)
            # Анализ спектра
            amps = np.abs(coeffs) / self.window_size
            
            # Выбор значимых компонент
            if self.use_adaptive_components:
                dominant_idx = self._select_components_by_energy(amps, freqs)
            else:
                if self.n_components is not None:
                    dominant_idx = np.argsort(amps)[-self.n_components:][:: -1]
                else:
                    dominant_idx = self._select_components_by_energy(amps, freqs)
                #n_comp = self._select_components_by_energy(amps, freqs)
            #else:
                #n_comp = self.n_components or self._select_components_by_energy(amps, freqs)
            
            # Сохранение доминирующих компонент
            #dominant_idx = np.argsort(amps)[- n_comp:][:: -1]
            
            self.window_coefficients_.append(coeffs)
            self.window_frequencies_.append(freqs)
            self.window_positions_.append(position)
            self.dominant_frequencies_.append(freqs[dominant_idx])
            self.dominant_amplitudes_.append(amps[dominant_idx])
        
        # Преобразование в numpy массивы
        self.window_coefficients_ = np.array(self.window_coefficients_)
        self.window_frequencies_ = np.array(self.window_frequencies_)
        self.window_positions_ = np.array(self.window_positions_)
        self.dominant_frequencies_ = np.array(self.dominant_frequencies_, dtype = object)
        print(self.dominant_frequencies_)
        self.dominant_amplitudes_ = np.array(self.dominant_amplitudes_, dtype = object)

        '''
        # Генерация параметров для реализации оконного преобразования Фурье
        self.window_size = STFFT.build_window_size(self.series)
        self.overlap = STFFT.overlap_selections(self.window_size)
        self.step_size = STFFT.calculate_step_size(self.window_size, self.overlap)
        self.sampling_rate = STFFT.calculate_sampling_rate(n, date_start, date_stop)

        # Количество окон, которые будут созданы при скользящем оконном анализе ВР. 
        # Это число определяет, сколько отдельных преобразований Фурье будет выполнено.
        n_windows = int((n - self.window_size) // self.step_size + 1)
        
        # Инициализация массивов
        self.window_coefficients_ = []
        self.window_frequencies_ = []
        self.window_positions_ = []
        self.dominant_frequencies_ = []
        self.dominant_amplitudes_ = []
        
        # Создание оконной функции
        window_func = self._create_window_function()
        
        print(f"Анализ {n_windows} окон...")
        
        for i in range(n_windows):
            start = int(i * self.step_size)
            end = int(start + self.window_size)
            position = start + self.window_size // 2
            
            # Обработка окна
            window_data = self.series[start: end] * window_func
            
            # FFT для окна
            coeffs = rfft(window_data)
            freqs = rfftfreq(self.window_size, 1 / self.sampling_rate)
            
            # Анализ спектра
            amps = np.abs(coeffs) / self.window_size
            
            # Выбор значимых компонент
            if self.use_adaptive_components:
                n_comp = self._select_components_by_energy(amps, freqs)
            else:
                n_comp = self.n_components or self._select_components_by_energy(amps, freqs)
            
            # Сохранение доминирующих компонент
            dominant_idx = np.argsort(amps)[- n_comp:][:: -1]
            
            self.window_coefficients_.append(coeffs)
            self.window_frequencies_.append(freqs)
            self.window_positions_.append(position)
            self.dominant_frequencies_.append(freqs[dominant_idx])
            self.dominant_amplitudes_.append(amps[dominant_idx])
        
        # Преобразование в numpy массивы
        self.window_coefficients_ = np.array(self.window_coefficients_)
        self.window_frequencies_ = np.array(self.window_frequencies_)
        self.window_positions_ = np.array(self.window_positions_)
        self.dominant_frequencies_ = np.array(self.dominant_frequencies_, dtype = object)
        self.dominant_amplitudes_ = np.array(self.dominant_amplitudes_, dtype = object)
        '''
    
    def _auto_trend_window(self):
        """
            Автоматический выбор окна для тренда
        """
        return min(30, max(5, len(self.series) // 10))


    def _extract_trend(self, series):
        """
            Извлечение тренда с помощью скользящего среднего
        """
        if not self.use_trend or len(series) < self.trend_window:
            return np.zeros_like(series), series.copy()
        
        # Скользящее среднее
        trend = np.convolve(series, np.ones(self.trend_window)/self.trend_window, mode='same')
        
        # Корректировка краев
        half_window = self.trend_window // 2
        trend[:half_window] = trend[half_window]
        trend[-half_window:] = trend[-half_window-1]
        
        detrended = series - trend
        return trend, detrended


    def _forecast_trend(self, n_steps):
        """
            Прогнозирование тренда с помощью скользящего среднего + экстраполяции
        """
        if self.trend_component_ is None or not self.use_trend:
            return np.zeros(n_steps)
        
        # Берем последние значения тренда для экстраполяции
        last_trend_values = self.trend_component_[-self.trend_window:]
        
        # Простая экстраполяция: средний прирост
        if len(last_trend_values) > 1:
            increments = np.diff(last_trend_values)
            avg_increment = np.mean(increments)
            
            # Прогноз тренда
            trend_forecast = np.zeros(n_steps)
            last_value = last_trend_values[-1]
            
            for i in range(n_steps):
                trend_forecast[i] = last_value + avg_increment * (i + 1)
            
            return trend_forecast
        else:
            return np.full(n_steps, last_trend_values[-1])

    
    def _select_components_by_energy(self, amplitudes, frequencies, energy_threshold = 0.95):
        """
            Выбор компонент основанный на доле энергии которую они объясняют.
            
            Parameters:
                amplitudes: амплитуды гармоник (отсортированные)
                energy_threshold: порог кумулятивной энергии (0-1)
            Returns:
                n_components: количество значимых компонент
        """
        # Сортируем по энергии
        # Сортируем амплитуды по убыванию
        sorted_idx = np.argsort(amplitudes)[::-1]
        sorted_amps = amplitudes[sorted_idx]
        sorted_freqs = frequencies[sorted_idx]
        
        # Исключаем очень низкочастотные компоненты (тренд)
        mask = sorted_freqs > 0.01  # исключаем частоты близкие к 0
        filtered_amps = sorted_amps[mask]
        filtered_freqs = sorted_freqs[mask]
        filtered_idx = sorted_idx[mask]
        
        # Вычисляем энергию (квадраты амплитуд)
        energies = filtered_amps ** 2
        total_energy = np.sum(energies)
        if total_energy > 0:
            # Кумулятивная энергия
            cumulative_energy = np.cumsum(energies) / total_energy
            # Находим где достигается порог
            n_components = np.argmax(cumulative_energy >= energy_threshold) + 1
        else:
            n_components = min(5, len(filtered_amps))
        return filtered_idx[:n_components]
    


    def predict(self, n_steps, method = 'smart', **kwargs):
        """
            Улучшенный predict с поддержкой обоих режимов
            
            Parameters:
                n_steps: количество шагов прогноза
                window_method: стратегия прогнозирования
                method: метод прогнозирования
                    'smart' - умный выбор based на обучении
                    'windowed' - оконные методы
                    'basic' - базовый метод родителя
        """
        if (self.window_coefficients_ is None) or len(self.window_coefficients_) == 0:
            raise ValueError('Оконные коэффициенты недоступны! Убедитесь, что модель обучена.')
        if not hasattr(self, 'coefficients_') and not self.window_trained:
            raise ValueError("Модель не обучена. Сначала вызовите fit()")
        if method == 'smart':
            return self._windowed_predict(n_steps)
        elif method == 'windowed':
            return self._windowed_predict(n_steps, kwargs.get('window_method', 'weighted'))
        else:
            raise ValueError("Доступные методы: 'smart', 'windowed'")


    def _windowed_predict(self, n_steps, window_method = 'evolutionary'):
        """
            Оконное прогнозирование с разными стратегиями
        """
        if window_method == 'weighted':
            return self._weighted_window_forecast(n_steps)
        elif window_method == 'evolutionary':
            return self._evolutionary_forecast(n_steps)
        elif window_method == 'last_window':
            return self._last_window_forecast(n_steps)
        else:
            raise ValueError("Неизвестный оконный метод")

    '''
    def _forecast_from_single_window(self, window_idx, n_steps):
        """
            Функция создаёт прогноз ВР на основе спектральных характеристик одного конкретного окна.
        """
        # Извлечение комплексных коэффициентов Фурье для указанного окна
        coeffs = self.window_coefficients_[window_idx]
        # Извлечение частот для указанного окна
        freqs = self.window_frequencies_[window_idx]
        # Создание временной оси для прогноза. Начинается с конца исходного сигнала и продолжается на n_steps вперед
        t_forecast = np.arange(self.signal_length_, self.signal_length_ + n_steps)
        forecast = np.zeros(n_steps)
        
        for i in range(len(coeffs)):
            # Фильтрация нулевых/незначимых компонент
            if np.abs(coeffs[i]) > 0:
                # Частота в Гц
                freq = freqs[i]
                amp = np.abs(coeffs[i]) / self.window_size
                phase = np.angle(coeffs[i])
                forecast += amp * np.cos(2 * np.pi * freq * t_forecast + phase)
        return forecast
    '''
    

    def _forecast_from_single_window(self, window_idx, n_steps):
        coeffs = self.window_coefficients_[window_idx]
        # Адаптивный выбор типа частот
        if self.sampling_rate < 0.001:
            # Используем нормализованные частоты для низкого sampling_rate
            freqs = np.fft.fftfreq(self.window_size, d = 1.0)[:len(coeffs)]
            time_factor = 1.0  # нормализованное время
        else:
            # Используем реальные частоты
            freqs = self.window_frequencies_[window_idx]
            time_factor = 1.0 / self.sampling_rate  # реальное время
        
        t_forecast = np.arange(n_steps) * time_factor
        forecast = np.zeros(n_steps)
        
        # Используем только значимые компоненты (топ-N по амплитуде)
        amplitudes = np.abs(coeffs) / self.window_size
        significant_indices = np.argsort(amplitudes)[-self.n_components:][::-1]
        
        for i in significant_indices:
            if amplitudes[i] > 1e-10:
                freq = freqs[i]
                amp = amplitudes[i]
                phase = np.angle(coeffs[i])
                forecast += amp * np.cos(2 * np.pi * freq * t_forecast + phase)
        return forecast
    

        
    def _weighted_window_forecast(self, n_steps):
        """
            Реализует взвешенное прогнозирование ВР комбинируя прогнозы из нескольких последних окон анализа 
            с разными весами важности. 
            Args:
                n_steps: количество шагов прогноза.
            Returns:
                forecast:
        """
        # 1. Прогноз детрендированной компоненты (циклические паттерны)
        n_last = min(3, len(self.window_coefficients_))
        weights = np.array([0.5, 0.3, 0.2][:n_last])
        weights /= weights.sum()
        
        detrended_forecast = np.zeros(n_steps)
        for i, weight in enumerate(weights):
            window_idx = -n_last + i
            window_forecast = self._forecast_from_single_window(window_idx, n_steps)
            detrended_forecast += weight * window_forecast
        
        # 2. Прогноз тренда
        trend_forecast = self._forecast_trend(n_steps)
        
        # 3. Комбинируем
        forecast_denorm = self._denorm_forecast(detrended_forecast)
        total_forecast = forecast_denorm + trend_forecast
        return total_forecast

        '''
        # Определение количества используемых окон. Берется минимум 3 из общего кол-ва. Если n_last = 2, используем 2 окна.
        n_last = min(3, len(self.window_coefficients_))
        # Генерация весов. Веса нормализуются так, чтобы их сумма равнялась 1. 
        # 0.5 - самые актуальные данные, наибольший вес; 0.3 - предпоследнее окно; 0.2 - третье окно с конца.
        weights = np.array([0.5, 0.3, 0.2][: n_last])
        weights /= weights.sum()
        
        # Инициализация массива с прогнозом
        forecast = np.zeros(n_steps)
        for i, weight in enumerate(weights):
            # Определение индекса окна
            window_idx = -n_last + i
            # Прогноз от отдельного окна
            window_forecast = self._forecast_from_single_window(window_idx, n_steps)
            # Взвешенное суммирование
            forecast += weight * window_forecast
        return forecast
        '''

    
    def _get_component_history(self, comp_idx, n_history = 5):
        """
            Получение истории параметров компоненты
        """
        freqs, amps = [], []
        for window_freqs, window_amps in zip(self.dominant_frequencies_[-n_history:], self.dominant_amplitudes_[-n_history:]):
            # Проверяется, существует ли компонента с и ндексом comp_idx в текущем окне
            if comp_idx < len(window_freqs):
                # Извлечение частоты и амплитуды из текущего окна
                freqs.append(window_freqs[comp_idx])
                amps.append(window_amps[comp_idx])
        return freqs, amps

    
    def _get_phase_history(self, comp_idx, n_history = 5):
        """
            Собирает историю фазовых углов для указанной гармонической компоненты из нескольких последних окон.
            Фаза - это критически важный параметр для точного прогнозирования, определяющий "сдвиг" гармонического сигнала во времени.
            Фазы измеряются в радианах!
            Args:
                comp_idx: индекс компоненты
                n_history: количество последних окон
            Returns:
                phases: список фаз в радианах
        """
        phases = []
        # Берем последние n_history окон, создаем отрицательные индексы
        for window_idx in range(-n_history, 0):
            # Преобразование отрицательного индекса в положительный
            window_idx_adj = window_idx if window_idx >= 0 else len(self.window_coefficients_) + window_idx
            # Проверяем что компонента существует в этом окне
            if comp_idx < len(self.dominant_frequencies_[window_idx_adj]):
                # Находим соответствующий коэффициент Фурье
                target_freq = self.dominant_frequencies_[window_idx_adj][comp_idx]   # частота компоненты из доминирующих частот
                freqs = self.window_frequencies_[window_idx_adj]                     # полный спектр частот для окна
                coeffs = self.window_coefficients_[window_idx_adj]                   # комлпексные коэффициенты Фурье для окна
                # Ищем ближайшую частоту
                freq_idx = np.argmin(np.abs(freqs - target_freq))
                # Извлекаем фазу
                phase = np.angle(coeffs[freq_idx]) # извлечение фазового угла из комплексного числа
                phases.append(phase)
        # Устраняем скачки через 2π для непрерывности. Фаза создается в диапазоне [-pi; pi], что создает искусственные разрывы
        if len(phases) > 1:
            # Если разница между соседними фазами больше pi, добавляем ±2pi
            phases = np.unwrap(phases)
        return phases

    '''    
    def _evolutionary_forecast(self, n_steps):
        """
            Реализует эволюционное прогнозирование - метод отслеживает и экстраполирует изменения спектральных параметров
            (частот и амплитуд) во времени, создавая прогноз основанный на трендах эволюции гармонических компонент.
        """
        # Определение кол-ва компонент. Берется последнее окно анализа и определяется кол-во гармонических компонент в этом окне.
        n_components = len(self.dominant_frequencies_[-1])
        # Инициализация массива с прогнозом
        forecast = np.zeros(n_steps)
        # Создание временной оси для прогноза
        t_forecast = np.arange(self.signal_length_, self.signal_length_ + n_steps)
        
        for comp_idx in range(n_components):
            # Сбор истории параметров для каждой компоненты
            freqs, amps = self._get_component_history(comp_idx)
            phases = self._get_phase_history(comp_idx)
            # Проверка достаточности данных. Нужно минимум 3 точки для линейной регрессии. Если данных меньше, компонента скипается.
            if len(freqs) > 2:
                # Экстраполяция трендов
                time_idx = np.arange(len(freqs))             # индекс окна
                freq_trend = np.polyfit(time_idx, freqs, 1)  # линейная регрессия для частот  
                amp_trend = np.polyfit(time_idx, amps, 1)    # линейная регрессия для амплитуд 
                phase_trend = np.polyfit(time_idx, phases, 1)# линейная регрессия для фазы 
                
                # Прогноз параметров в будущее
                future_idx = np.arange(len(freqs), len(freqs) + n_steps)
                future_freqs = np.polyval(freq_trend, future_idx)
                future_amps = np.polyval(amp_trend, future_idx)
                future_phases = np.polyval(phase_trend, future_idx)
                
                # Добавление гармонической компоненты с учетом фазы
                for i in range(n_steps):
                    forecast[i] += future_amps[i] * np.cos(2 * np.pi * future_freqs[i] * t_forecast[i] + future_phases[i])
        return forecast
    '''
    
    
    def _evolutionary_forecast(self, n_steps):
        """
         Эволюционный прогноз с трендом
        """
        # Детрендированная компонента
        detrended_forecast = self._evolutionary_forecast_detrended(n_steps)
        forecast_denorm = self._denorm_forecast(detrended_forecast)
        # Трендовая компонента
        trend_forecast = self._forecast_trend(n_steps)
        return forecast_denorm + trend_forecast


    def _evolutionary_forecast_detrended(self, n_steps):
        """
         Эволюционный прогноз только для детрендированной компоненты
        """
        # Определение кол-ва компонент. Берется последнее окно анализа и определяется кол-во гармонических компонент в этом окне.
        n_components = len(self.dominant_frequencies_[-1])
        # Инициализация массива с прогнозом
        forecast = np.zeros(n_steps)
        # Создание временной оси для прогноза
        t_forecast = np.arange(self.signal_length_, self.signal_length_ + n_steps)
        
        for comp_idx in range(n_components):
            # Сбор истории параметров для каждой компоненты
            freqs, amps = self._get_component_history(comp_idx)
            phases = self._get_phase_history(comp_idx)
            # Проверка достаточности данных. Нужно минимум 3 точки для линейной регрессии. Если данных меньше, компонента скипается.
            if len(freqs) > 2:
                # Экстраполяция трендов
                time_idx = np.arange(len(freqs))             # индекс окна
                freq_trend = np.polyfit(time_idx, freqs, 1)  # линейная регрессия для частот  
                amp_trend = np.polyfit(time_idx, amps, 1)    # линейная регрессия для амплитуд 
                phase_trend = np.polyfit(time_idx, phases, 1)# линейная регрессия для фазы 
                
                future_idx = np.arange(len(freqs), len(freqs) + n_steps)
                future_freqs = np.polyval(freq_trend, future_idx)
                future_amps = np.polyval(amp_trend, future_idx)
                future_phases = np.polyval(phase_trend, future_idx)
                
                for i in range(n_steps):
                    forecast[i] += future_amps[i] * np.cos(2 * np.pi * future_freqs[i] * t_forecast[i] + future_phases[i])
        return forecast


    
    def _last_window_forecast(self, n_steps):
        """
            Прогноз по последнему окну
        """
        forecast = self._forecast_from_single_window(-1, n_steps)
        return forecast


    def _denorm_forecast(self, forecast):
        """
            Обратная обработка прогноза
        """
        if not self.is_fitted:
            raise ValueError("Scaler не обучен. Сначала вызовите fit()")
        # Денормализация
        forecast_denorm = self.scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
        return forecast_denorm

    def _retrend_forecast(self, forecast, last_trend, n_steps):
        """
            Восстановление тренда в прогнозе
        """
        x_forecast = np.arange(n_steps)
        trend_component = np.polyval(last_trend, x_forecast + self.window_size)
        return forecast + trend_component


class PipeLine:
    def __init__():
        pass

    @staticmethod
    def calculate_share_not_found_programs(new_df, df_hist):
        """
            Функция для расчета прогноза доли для программ, для которых не было найдено похожей программы.
            Функция генерит последние 4 недели и ищет сначала делает поиск по слоту. Считается средняя доля программ для конкретного слота, 
            в котором шла программа. Если не было найдено совпадение по слоту, то считается среднее за последние 4 недели.
        """
        new_df_ = new_df.reset_index(drop = True)
        for i in range(len(new_df_)):
            start_date = new_df_.iloc[i]['Дата']
            slot = new_df_.iloc[i]['Время выхода']
            
            dates = Dates_Operations.get_last_4_weeks(start_date)
            timestamp_dates = [pd.Timestamp(d) for d in dates]
            df_dates = pd.DataFrame(timestamp_dates, columns = ['Дата'])
            #Join дат и исторического DataFrame
            merged = pd.merge(df_dates, df_hist, on = 'Дата', how = 'left')
            #Отбор по слоту
            slot_df = merged[merged['Время выхода'] == slot]
            #Если нашлась какая-то программа по тому же слоту
            if len(slot_df) != 0:
                share_mean = np.mean(list(slot_df['Share']))
                new_df_.at[i, 'Forecast'] = share_mean
            #Если НЕ нашлась какая-то программа по тому же слоту
            elif len(slot_df) == 0:
                share_mean = np.mean(list(merged['Share']))
                new_df_.at[i, 'Forecast'] = share_mean
        return new_df_