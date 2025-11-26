class Task:
    """
    state:
    0: unexecuted;
    1: executing;
    2: executed;
    3: calculated utility
    local: strategy = 1
    mec: (strategy > 1) && (strategy <= size(all_MECs, 2) + 1)
    uav: (strategy > size(all_MECs, 2) + 1) && (strategy <= size(all_MECs, 2) + 1 + size(all_UAVs, 2));
    lang: (strategy >= sum_strategy + 1) && (strategy <= 2 * sum_strategy)
    lang: (strategy > size(all_MECs, 2) + 1 + size(all_UAVs, 2)) && (strategy <= 2 * (size_mec + 1)))
    """

    def __init__(self):
        self.state = 0
        self.size = 0
        self.cpu_per_bit = 0
        self.cpu = 0
        self.max_delay = 0
        self.task_arrival_rate = 0
        self.remain_delay = 0
        self.max_delay_static = 0
        self.preference_list = []
        self.discount_factor_bargain = 1

    # add to preference list
    def add_preference(self, new_preference_struct):
        self.preference_list.append(new_preference_struct)

    # sort preference -- vehicle utility
    def sort_preference_list(self):
        self.preference_list.sort(key=lambda x: x['vutility'], reverse=True)

    # delete preference by mec_id
    def delete_preference(self, delete_mid):
        self.preference_list = [preference for preference in self.preference_list if preference['mid'] != delete_mid]

    def clear_preference_list(self):
        self.preference_list = []
