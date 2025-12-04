import pandas as pd
import numpy as np
import datetime as dt
import sys
import xlsxwriter
from datetime import datetime, timedelta
import pymorphy3 as pmrph
import time
import threading
from threading import Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
import OMA_tools
from OMA_tools.io_data.operations import File, Table, Dict_Operations
from OMA_tools.io_data.dates import Dates_Operations
import os
import re
import json
import time
import openpyxl
import copy
from IPython.display import JSON
from mediascope_api.core import net as mscore
from mediascope_api.mediavortex import tasks as cwt
from mediascope_api.mediavortex import catalogs as cwc

import warnings
warnings.filterwarnings('ignore')

# Настраиваем отображение
pd.set_option('display.max_columns', None)

# Создаем объекты для работы с TVI API
mnet = mscore.MediascopeApiNetwork()
mtask = cwt.MediaVortexTask()
cats = cwc.MediaVortexCats()


#Класс, который мьютит все принты в консоли
class WrapperNoPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout



class BaseDataService:
    """
        Базовый класс для устранения дублирования кода
    """
    @staticmethod
    def _build_common_params(date_filter, company_filter, basedemo_filter, regions_id, targets,
                           time_filter, statistics, slices, sortings, options,
                           location_filter, weekday_filter, daytype_filter, 
                           targetdemo_filter, add_city_to_basedemo_from_region, add_city_to_targetdemo_from_region):
        """
            Построение общих параметров для задач
        """
        # ПОЛНОЕ КОПИРОВАНИЕ ВСЕХ ПАРАМЕТРОВ
        safe_regions_id = copy.deepcopy(regions_id)
        safe_targets = copy.deepcopy(targets)

        safe_params = {
            'date_filter': copy.deepcopy(date_filter) if date_filter is not None else None,
            'company_filter': copy.deepcopy(company_filter) if company_filter is not None else None,
            'basedemo_filter': copy.deepcopy(basedemo_filter) if basedemo_filter is not None else None,
            'regions_id': copy.deepcopy(regions_id) if regions_id is not None else {},
            'time_filter': copy.deepcopy(time_filter) if time_filter is not None else None,
            'statistics': copy.deepcopy(statistics) if statistics is not None else None,
            'slices': copy.deepcopy(slices) if slices is not None else None,
            'sortings': copy.deepcopy(sortings) if sortings is not None else None,
            'options': copy.deepcopy(options) if options is not None else None,
            'location_filter': copy.deepcopy(location_filter) if location_filter is not None else None,
            'weekday_filter': copy.deepcopy(weekday_filter) if weekday_filter is not None else None,
            'daytype_filter': copy.deepcopy(daytype_filter) if daytype_filter is not None else None,
            'targetdemo_filter': copy.deepcopy(targetdemo_filter) if targetdemo_filter is not None else None
        }

        # 1. Случай, когда задан словарь с regions_id
        if regions_id is not None and basedemo_filter is not None and targets is None:
            tasks = []
            for reg_id, reg_name in safe_regions_id.items():
                current_company_filter = safe_params['company_filter']
                
                if current_company_filter is not None:
                    current_company_filter = current_company_filter + f' AND regionId IN ({reg_id})'
                else:
                    current_company_filter = f'regionId IN ({reg_id})'
                
                tasks.append({
                        'project_name': reg_name,
                        'task': mtask.send_timeband_task(
                            mtask.build_timeband_task(
                                date_filter = safe_params['date_filter'], 
                                weekday_filter = safe_params['weekday_filter'], 
                                daytype_filter = safe_params['daytype_filter'], 
                                company_filter = current_company_filter, 
                                time_filter = safe_params['time_filter'], 
                                basedemo_filter = safe_params['basedemo_filter'], 
                                targetdemo_filter = safe_params['targetdemo_filter'],
                                location_filter = safe_params['location_filter'],
                                slices = safe_params['slices'], 
                                sortings = safe_params['sortings'],
                                statistics = safe_params['statistics'],
                                options = safe_params['options'],
                                add_city_to_basedemo_from_region = add_city_to_basedemo_from_region,
                                add_city_to_targetdemo_from_region = add_city_to_targetdemo_from_region)
                                )
                            })
                time.sleep(2)        
            return tasks

        # 2. Случай, когда задан словарь с ЦА targets
        elif targets is not None and regions_id is None and basedemo_filter is None:
            tasks = []
            for target, syntax in safe_targets.items():
                basedemo_filter = syntax

                tasks.append({
                        'project_name': target,
                        'task': mtask.send_timeband_task(
                            mtask.build_timeband_task(
                                date_filter = safe_params['date_filter'], 
                                weekday_filter = safe_params['weekday_filter'], 
                                daytype_filter = safe_params['daytype_filter'], 
                                company_filter = safe_params['company_filter'], 
                                time_filter = safe_params['time_filter'], 
                                basedemo_filter = safe_params['basedemo_filter'], 
                                targetdemo_filter = safe_params['targetdemo_filter'],
                                location_filter = safe_params['location_filter'],
                                slices = safe_params['slices'], 
                                sortings = safe_params['sortings'],
                                statistics = safe_params['statistics'],
                                options = safe_params['options'],
                                add_city_to_basedemo_from_region = add_city_to_basedemo_from_region,
                                add_city_to_targetdemo_from_region = add_city_to_targetdemo_from_region)
                                )
                            })
                time.sleep(2)
            return tasks
        
        # 2. Случай, когда задан словарь с ЦА targets
        elif targets is not None and regions_id is not None and basedemo_filter is None:
            if type(regions_id) == int:
                current_company_filter = safe_params['company_filter'] + f' AND regionId IN ({regions_id})' 

                tasks = []
                for target, syntax in safe_targets.items():
                    basedemo_filter = syntax

                    tasks.append({
                            'project_name': target,
                            'task': mtask.send_timeband_task(
                                mtask.build_timeband_task(
                                    date_filter = safe_params['date_filter'], 
                                    weekday_filter = safe_params['weekday_filter'], 
                                    daytype_filter = safe_params['daytype_filter'], 
                                    company_filter = current_company_filter, 
                                    time_filter = safe_params['time_filter'], 
                                    basedemo_filter = basedemo_filter, 
                                    targetdemo_filter = safe_params['targetdemo_filter'],
                                    location_filter = safe_params['location_filter'],
                                    slices = safe_params['slices'], 
                                    sortings = safe_params['sortings'],
                                    statistics = safe_params['statistics'],
                                    options = safe_params['options'],
                                    add_city_to_basedemo_from_region = add_city_to_basedemo_from_region,
                                    add_city_to_targetdemo_from_region = add_city_to_targetdemo_from_region)
                                    )
                                })
                    time.sleep(2)
                return tasks
        
        # 3. Случай, когда задан только basedemo_filter, а targets и regions_id не заданы
        elif basedemo_filter is not None and regions_id is None and targets is None:
            task = mtask.build_timeband_task(
                                date_filter = safe_params['date_filter'], 
                                weekday_filter = safe_params['weekday_filter'], 
                                daytype_filter = safe_params['daytype_filter'], 
                                company_filter = safe_params['company_filter'], 
                                time_filter = safe_params['time_filter'], 
                                basedemo_filter = safe_params['basedemo_filter'], 
                                targetdemo_filter = safe_params['targetdemo_filter'],
                                location_filter = safe_params['location_filter'],
                                slices = safe_params['slices'], 
                                sortings = safe_params['sortings'],
                                statistics = safe_params['statistics'],
                                options = safe_params['options'],
                                add_city_to_basedemo_from_region = add_city_to_basedemo_from_region,
                                add_city_to_targetdemo_from_region = add_city_to_targetdemo_from_region)
            return task


    @staticmethod
    def _execute_tasks(tasks):
        """
            Общая логика выполнения задач
        """
        if type(tasks) == list:
            tsks = mtask.wait_task(tasks)
            
            # Получаем результат
            results = []
            failed_tasks_ids: list[str] = []
            for t in tasks:
                tsk = t['task'] 

                # HOTFIX #
                outcome = mtask.get_status({
                    'taskId': tsk['taskId']
                })
                #print(outcome)

                if outcome.get('taskStatus').lower() in ['cancelled', 'failed']:
                    print('ERROR IN TASK!!!!!')
                    failed_tasks_ids.append(tsk['taskId'])
                    continue
                # END HOTFIX #

                df_result = mtask.result2table(mtask.get_result(tsk), project_name = t['project_name'])

                if df_result is None or len(df_result) < 1:
                    print('EMPTY RESULT!!!!!!!')
                    failed_tasks_ids.append(tsk['taskId'])
                    continue

                results.append(df_result)
            
            # Повторный запуск неудачных задач
            if failed_tasks_ids and len(failed_tasks_ids) > 0:
                mtask.restart_tasks(failed_tasks_ids)

                for tsk_id in failed_tasks_ids:
                    tsk = {
                        'taskId': tsk_id
                    }
                    tsk = mtask.wait_task(tsk)
                    df_result = mtask.result2table(mtask.get_result(tsk), project_name = tsk['project_name'])        
                    results.append(df_result)
                
            return pd.concat(results, ignore_index = True)
        
        elif type(tasks) == str:
            # Отправляем задание на расчет и ждем выполнения
            task_timeband = mtask.wait_task(mtask.send_timeband_task(tasks))

            # Получаем результат
            df = mtask.result2table(mtask.get_result(task_timeband), project_name = 'Total. Ind')
            return df