import random
from utils.Task import Task  # 导入 Task 类
from utils.Base import Base

def InitializeTask(base):
    all_tasks = []

    def InitializeTaskParams(id):
        single_task = Task()
        single_task.id = id
        single_task.state = 0
        single_task.size = 7* 10 ** 6
        single_task.cpu_per_bit = 500 + 1000 * random.random()
        single_task.cpu = single_task.cpu_per_bit * single_task.size
        single_task.max_delay = 0.5 + 5.5 * random.random()
        single_task.task_arrival_rate = 0.6 + 0.4 * random.random()
        single_task.remain_delay = single_task.max_delay
        single_task.max_delay_static = single_task.max_delay
        single_task.preference_list = {'mid': [], 'vutility': [], 'req_allo': [], 'price': [], 'mutility': []}

        single_task.discount_factor_bargain = base.discount_factor_bargain

        return single_task

    task_num = 1

    for i in range(base.all_Vehicles_num):
        single_task = InitializeTaskParams(i)
        all_tasks.append(single_task)
    return all_tasks
