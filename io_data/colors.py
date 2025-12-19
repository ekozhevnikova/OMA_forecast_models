import os
import re

class Color:
    # Существующие константы
    BOLD = '\033[1m'
    BLUE = '\x1b[38;2;0;0;255m'
    GREEN = '\x1b[38;2;0;138;69m'
    YELLOW = '\x1b[38;2;255;197;0m'
    ORANGE = '\x1b[38;2;255;86;3m'
    PURPLE = '\x1b[38;2;243;86;255m'
    VIOLET = '\x1b[38;2;175;0;255m'
    RED = '\x1b[38;2;197;7;0m'
    END = '\033[0m'
    UNDERLINE = '\033[4m'
    
    # Новые методы для работы с RGB
    @classmethod
    def rgb(cls, r, g, b, background = False):
        """
        Создает ANSI-код из RGB значений.
        
        Args:
            r, g, b: значения от 0 до 255
            background: если True, применяется к фону
        """
        mode = 48 if background else 38
        return f'\033[{mode};2;{r};{g};{b}m'
    

    @classmethod
    def center_text(cls, text, styles=''):
        """
            Центрирует текст с учетом ANSI кодов
            
            Args:
                text: текст для центрирования
                styles: строка со стилями (ANSI коды)
        """
        # Получаем ширину терминала
        try:
            width = os.get_terminal_size().columns
        except:
            width = 80
        
        # Убираем ANSI коды для расчета длины
        clean_text = re.sub(r'\033\[[0-9;]*m', '', text)
        
        # Рассчитываем отступы
        text_len = len(clean_text)
        if text_len >= width:
            return styles + text + cls.END  # Используем cls.END вместо Color.END
        
        left_pad = (width - text_len) // 2
        return ' ' * left_pad + styles + text + cls.END  # Используем cls.END