import numpy as np
import pandas as pd
from scipy.fft import fft, rfft, rfftfreq, ifft, fftfreq
import warnings
warnings.filterwarnings('ignore')


class STFT:
    """
        Класс для реализации оконного преобразования Фурье.
    """
    def __init__(self, series, window_type = 'hann'):
        self.series = series
        self.window_type = window_type        # тип оконой функции


        # Параметры, которые задаются позже
        self.window_size = None        # размер окна
        self.overlap = None            # размер окна
        self.step_size = None          # размер окна
        self.sampling_rate = None      # размер окна
        self.signal_length = None
        self.n_components = None

        self.window_coefficients_ = None
        self.window_frequencies_ = None
        self.window_positions_ = None
        self.dominant_frequencies_ = None
        self.dominant_amplitudes_ = None

        self.window_trained = False

    @staticmethod
    def round_half_up(x):
                return int(x + 0.5)

    def stft_params(self, date_start, date_stop):
        """
            Метод для генерации входных параметров для оконного преобразования Фурье.
        """
        # Длина ВР
        self.signal_length = len(self.series)

        # Генерация размера окна
        #Для коротких ВР
        if len(self.series) < 500:
            window_size_lower = float(0.25 * len(self.series))
            window_size_upper = float(0.5 * len(self.series))
            #Среднее между верхним и нижним пределом
            self.window_size = STFT.round_half_up(float(0.5 * (window_size_lower + window_size_upper)))
            print(f'Сигнал короткий! Оптимальный размер окна в промежутке {window_size_lower} - {window_size_upper}.')
            print(f'Выбираю среднее: {self.window_size}.')

        #Для средних ВР
        elif len(self.series) >= 500 and len(self.series) < 5000:
            window_size_lower = float(0.1 * len(self.series))
            window_size_upper = float(0.2 * len(self.series))
            #Среднее между верхним и нижним пределом
            self.window_size = STFT.round_half_up(float(0.5 * (window_size_lower + window_size_upper)))
            print(f'Сигнал средний! Оптимальный размер окна в промежутке {window_size_lower} - {window_size_upper}.')
            print(f'Выбираю среднее: {self.window_size}.')

        #Для больших ВР
        elif len(self.series) >= 5000:
            self.window_size = STFT.round_half_up(float(0.1 * len(self.series)))
            print(f'Сигнал длинный! Оптимальный размер окна {self.window_size}.')

        # Генерация процента перекрытия
        #Маленькие окна (С окном < 64 нельзя различить частоты ближе, чем ~15 Гц при типичных sampling_rate)
        if self.window_size < 64:
            self.overlap = 0.75
        #Большие окна
        elif self.window_size > 512:
            self.overlap = 0.6
        #Средние окна
        else:
            self.overlap = 0.75

        # Генерация шага перемещения окна
        #Если overlap указан в процентах (например, 75)
        if self.overlap > 1.0:
            overlap_ = self.overlap / 100
            self.step_size = STFT.round_half_up(float(self.window_size * (1.0 - overlap_)))
        #Если overlap указан в долях (например, 0.75)
        else:
            self.step_size = STFT.round_half_up(float(self.window_size * (1.0 - self.overlap)))


        # Генерация частоты дискретизации
        # Если временные метки доступны, используем их для точного расчета
        if hasattr(self.series, 'index') and isinstance(self.series.index, pd.DatetimeIndex):
            time_diffs = np.diff(self.series.index.astype(np.int64) // 10**9)  # разница в секундах
            if len(time_diffs) > 0:
                avg_interval = np.median(time_diffs)
                self.sampling_rate = 1.0 / avg_interval if avg_interval > 0 else 1.0
            else:
                total_days = (date_stop - date_start).days
                self.sampling_rate = self.signal_length / max(total_days, 1)
        else:
            total_days = (date_stop - date_start).days
            self.sampling_rate = self.signal_length / max(total_days, 1)

        '''
        # Для дневных данных используем дни как основную единицу
        total_days = (date_stop - date_start).days
        if total_days > 0:
            self.sampling_rate = STFT.round_half_up(float(self.signal_length / total_days))  # измерений в день
        else:
            self.sampling_rate = 1.0  # fallback
        # Если sampling_rate слишком мал, используем нормализованные частоты
        if self.sampling_rate < 0.001:
            print(f"Внимание: низкий sampling_rate ({self.sampling_rate:.6f}), используются нормализованные частоты")
        '''
        
        return self.signal_length, self.window_size, self.overlap, self.step_size, self.sampling_rate


    def _create_window_function(self):
        """
            Создание оконной функции.
            По умолчанию оконная функция Ханна. Но также мождно выбрать Блэкман, Прямоугольное окно.
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
        
    def _select_components_by_energy(self, amplitudes, frequencies, energy_threshold=0.95):
        """Упрощенный и стабильный выбор компонент"""
        # Сортируем по амплитуде
        sorted_idx = np.argsort(amplitudes)[::-1]
        sorted_amps = amplitudes[sorted_idx]
        
        # Базовый выбор по энергии - БЕЗ фильтрации низких частот
        energies = sorted_amps ** 2
        total_energy = np.sum(energies)
        
        if total_energy > 0:
            cumulative_energy = np.cumsum(energies) / total_energy
            n_components = np.argmax(cumulative_energy >= energy_threshold) + 1
            # Ограничиваем разумными пределами
            self.n_components = min(max(3, n_components), 20)
        else:
            self.n_components = 5
        
        return self.n_components
    
    '''
    def _select_components_by_energy(self, amplitudes, frequencies, energy_threshold = 0.95):
        """
        Выбор компонент основанный на доле энергии которую они объясняют.
        
        Parameters:
            amplitudes: амплитуды гармоник (отсортированные)
            energy_threshold: порог кумулятивной энергии (0 - 1)
        Returns:
            n_components: количество значимых компонент
        """
        # Сортируем амплитуды по убыванию
        sorted_idx = np.argsort(amplitudes)[::-1]
        sorted_amps = amplitudes[sorted_idx]
        sorted_freqs = frequencies[sorted_idx]
        
        # Исключаем очень низкочастотные компоненты (тренд)
        mask = sorted_freqs > 0.001  # исключаем частоты близкие к 0
        filtered_amps = sorted_amps[mask]
        filtered_freqs = sorted_freqs[mask]
        filtered_idx = sorted_idx[mask]
        min_components = max(3, int(0.1 * len(filtered_amps)))
        
        # Выбор по энергии
        energies = filtered_amps ** 2
        total_energy = np.sum(energies)
        if total_energy > 0:
            cumulative_energy = np.cumsum(energies) / total_energy
            n_components = np.argmax(cumulative_energy >= energy_threshold) + 1
            self.n_components = max(min_components, n_components)
            return self.n_components
        else:
            self.n_components = min_components
            return self.n_components
    '''
    

    def _windowed_fit(self, date_start, date_stop):
        """
            Оконное обучение с улучшенной предобработкой
        """
        print("=== ОКОННОЕ ОБУЧЕНИЕ ===")
        # 2. Оконное преобразование Фурье
        self.perform_windowed_analysis(date_start, date_stop)
        #print(f"Обучение завершено: {len(self.window_positions_)} окон")
        return self
    

    def fit(self, date_start, date_stop):
        """
            Улучшенный метод fit с поддержкой обоих режимов
            
            Parameters:
                target_column: название столбца с таргетом
        """
        self.series = np.array(self.series).flatten()
        self.window_trained = True
        return self._windowed_fit(date_start, date_stop)


    def perform_windowed_analysis(self, date_start, date_stop):
        """
            Выполнение оконного преобразования Фурье.
            Args:
                target_column: название столбца с таргетом.
                date_start: дата старта в формате Timestamp, нужно для расчета sampling_rate
                date_stop: дата конца в формате Timestamp, нужно для расчета sampling_rate
        """
        # 1. Инициализация параметров дла оконного анализа
        self.signal_length, self.window_size, self.overlap, self.step_size, self.sampling_rate = self.stft_params(date_start, date_stop)
        print(f'Выбраны следующие параметры для обучения: {self.signal_length}, {self.window_size}, {self.overlap}, {self.step_size}, {self.sampling_rate}')
        # 2. Оконный анализ (остальной код без изменений, но используем analysis_series)
        n_windows = int((self.signal_length - self.window_size) // self.step_size + 1)

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
            window_data = self.series[start: end] * window_func
            # FFT для окна
            coeffs = rfft(window_data)
            freqs = rfftfreq(self.window_size, 1 / self.sampling_rate)
            # Анализ спектра
            amps = np.abs(coeffs) / self.window_size
            
            # Выбор значимых компонент
            self.n_components = self._select_components_by_energy(amps, freqs)
            dominant_idx = np.argsort(amps)[- self.n_components:][:: -1]
            
            
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
    

    def predict(self, n_steps, window_method):
        """С проверкой данных"""
        if (self.window_coefficients_ is None) or len(self.window_coefficients_) == 0:
            raise ValueError('Оконные коэффициенты недоступны!')
        
        # Проверяем, что есть достаточное количество окон
        if len(self.window_coefficients_) < 2:
            print("Предупреждение: очень мало окон для анализа")
            return self._last_window_forecast(n_steps)
        
        # Диагностика
        self.debug_forecast_components(min(n_steps, 5))
        
        return self._windowed_predict(n_steps, window_method)

    '''
    def predict(self, n_steps, window_method):
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
        return self._windowed_predict(n_steps, window_method)
    '''
    

    def _windowed_predict(self, n_steps, window_method):
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


    def _forecast_from_single_window(self, window_idx, n_steps):
        """Исправленная версия прогноза из одного окна"""
        coeffs = self.window_coefficients_[window_idx]
        freqs = self.window_frequencies_[window_idx]
        
        # КРИТИЧЕСКИ ВАЖНО: правильная временная ось
        # Время должно начинаться с конца обучающего ряда
        t_forecast = np.arange(n_steps) / self.sampling_rate
        
        forecast = np.zeros(n_steps)
        amplitudes = np.abs(coeffs) / self.window_size
        
        # Берем топ компоненты
        significant_indices = np.argsort(amplitudes)[-self.n_components:][::-1]
        
        for i in significant_indices:
            if amplitudes[i] > 1e-10:  # избегаем численных ошибок
                freq = freqs[i]
                amp = amplitudes[i] * 2  # компенсация оконной функции
                phase = np.angle(coeffs[i])
                
                # Добавляем гармоническую компоненту
                forecast += amp * np.cos(2 * np.pi * freq * t_forecast + phase)
        
        return forecast

    '''
    def _forecast_from_single_window(self, window_idx, n_steps):
        """
            Функция создаёт прогноз ВР на основе спектральных характеристик одного конкретного окна.
        """
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
    '''
        
    
    def _weighted_window_forecast(self, n_steps, weights = [0.5, 0.3, 0.2]):
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
        weights = np.array(weights[:n_last])
        weights /= weights.sum()
        
        forecast = np.zeros(n_steps)
        for i, weight in enumerate(weights):
            window_idx = -n_last + i
            window_forecast = self._forecast_from_single_window(window_idx, n_steps)
            forecast += weight * window_forecast
        return forecast
    
    def _evolutionary_forecast(self, n_steps):
        """Упрощенный эволюционный прогноз"""
        forecast = np.zeros(n_steps)
        t_forecast = np.arange(n_steps) / self.sampling_rate
        
        # Используем только последние 3 окна для стабильности
        n_comps = len(self.dominant_frequencies_[-1])
        
        for comp_idx in range(min(n_comps, 10)):  # ограничиваем число компонент
            freqs, amps = self._get_component_history(comp_idx, n_history=3)
            
            if len(freqs) >= 2:
                # Простая линейная экстраполяция
                if len(freqs) == 2:
                    # Линейная интерполяция для 2 точек
                    freq_slope = freqs[1] - freqs[0]
                    amp_slope = amps[1] - amps[0]
                else:
                    # Линейная регрессия для 3+ точек
                    time_idx = np.arange(len(freqs))
                    freq_slope = (freqs[-1] - freqs[0]) / (len(freqs) - 1)
                    amp_slope = (amps[-1] - amps[0]) / (len(freqs) - 1)
                    future_freq = freqs[-1] + freq_slope
                future_amp = max(amps[-1] + amp_slope, 0)  # амплитуда неотрицательна
                
                # Берем фазу из последнего окна
                if comp_idx < len(self.dominant_frequencies_[-1]):
                    target_freq = self.dominant_frequencies_[-1][comp_idx]
                    freqs_full = self.window_frequencies_[-1]
                    coeffs_full = self.window_coefficients_[-1]
                    freq_idx = np.argmin(np.abs(freqs_full - target_freq))
                    phase = np.angle(coeffs_full[freq_idx])
                    
                    forecast += future_amp * np.cos(2 * np.pi * future_freq * t_forecast + phase)
        
        return forecast

    '''
    def _evolutionary_forecast(self, n_steps):
        """
            Эволюционный прогноз только для детрендированной компоненты
        """
        # Определение кол-ва компонент. Берется последнее окно анализа и определяется кол-во гармонических компонент в этом окне.
        n_comps = len(self.dominant_frequencies_[-1])
        # Инициализация массива с прогнозом
        forecast = np.zeros(n_steps)
        # Создание временной оси для прогноза
        t_forecast = np.arange(self.signal_length, self.signal_length + n_steps)
        
        for comp_idx in range(n_comps):
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


    def _last_window_forecast(self, n_steps):
        """
            Прогноз по последнему окну
        """
        forecast = self._forecast_from_single_window(-1, n_steps)
        return forecast
    
    # Добавьте этот метод для диагностики
    def debug_forecast_components(self, n_steps=10):
        """Отладочная информация о компонентах прогноза"""
        print("\n=== ДИАГНОСТИКА ПРОГНОЗА ===")
        print(f"Размер ряда: {len(self.series)}")
        print(f"Размер окна: {self.window_size}")
        print(f"Количество окон: {len(self.window_coefficients_)}")
        print(f"Sampling rate: {self.sampling_rate}")
        
        # Анализ последнего окна
        last_amps = self.dominant_amplitudes_[-1]
        last_freqs = self.dominant_frequencies_[-1]
        print(f"\nПоследнее окно - топ-5 компонент:")
        for i, (amp, freq) in enumerate(zip(last_amps[:5], last_freqs[:5])):
            print(f"  Компонента {i}: амплитуда={amp:.4f}, частота={freq:.4f}")
        
        # Тестовый прогноз
        test_forecast = self._forecast_from_single_window(-1, n_steps)
        print(f"\nТестовый прогноз (последнее окно): {test_forecast}")
        print(f"Диапазон прогноза: {np.min(test_forecast):.4f} до {np.max(test_forecast):.4f}")
        
        return test_forecast