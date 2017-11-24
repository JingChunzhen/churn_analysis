import numpy as np
import sqlite3
import numpy as np
import pickle
import re


def data_parser():
    '''
    针对每一个玩家的动作序列，制作相应的动作序列和标签
    '''
    conn = sqlite3.connect('./data/an.db2')
    c = conn.cursor()
    query_sql = "SELECT user_id, op, current_day, num_days_played, relative_timestamp \
        FROM maidian ORDER BY user_id, relative_timestamp"

    previous_relativetimestamp = None
    previous_userid = None
    previous_day = None
    previous_num_days_played = None
    
    user_ops = {}
    user_label = {}
    ops = []
    for row in c.execute(query_sql):        # 以上的处理导致动作数量为1的玩家纳入不进去
        user_id = row[0]
        op = row[1]
        current_day = row[2]
        num_days_played = row[3]
        # relative_timestamp = row[4]
        if previous_userid is None:
            ops = [op]            
        elif previous_userid == user_id:
            ops.append(op)            
        else:
            user_ops[previous_userid] = ops                
            if previous_num_days_played >= 1 and previous_num_days_played <= 3 and previous_day == previous_num_days_played:
                # 流失玩家
                user_label[user_id] = 1            
            else:      
                # 非流失玩家   
                user_label[user_id] = 0
        ops = [op]    #intervals
        previous_userid = user_id
        previous_day = current_day
        previous_num_days_played = num_days_played
    
    with open('./temp/slg_train.', 'wb') as f_out:
        pickle.dump(user_ops, f_out)
        pickle.dump(user_label, f_out)

def count_opintervals():
    '''
    得到每一个动作的时间间隔
    '''
    conn = sqlite3.connect('./data/an.db2')
    c = conn.cursor()
    query_sql = "SELECT user_id, op, current_day, num_days_played, relative_timestamp \
        FROM maidian ORDER BY user_id, relative_timestamp"

    previous_relativetimestamp = 0
    previous_userid = None
    previous_op = None
    intervals = []
    op_intervals = {}
    for row in c.execute(query_sql):
        '''
        针对每一个动作进行计算动作随后的时间间隔　
        '''
        user_id = row[0]
        op = row[1]
        current_day = row[2]
        num_days_played = row[3]
        relative_timestamp = row[4]
        interval = relative_timestamp - previous_relativetimestamp # 时间间隔 
        if previous_userid == user_id:
            if previous_op not in op_intervals:
                op_intervals[previous_op] = []
            op_intervals[previous_op].append(interval)          
        else:
            pass
        previous_userid = user_id
        previous_relativetimestamp = relative_timestamp
        previous_op = op
    return op_intervals

def churn_rate():
    '''
    计算每一个动作的点击人数和相应的流失人数
    bug 有一个地方在于 user_ops 和 user_label 中各有一个在对方的字典中不存在{979863553}    {262834177}
    '''
    with open('./temp/slg_train.', 'rb') as f_in:
        user_ops = pickle.load(f_in)
        user_label = pickle.load(f_in)

    print(len(user_label)) # 1071
    print(len(user_ops)) # 1071
    op_churn = {}
    op_categories = set()
    [op_categories.add(op) for userid, ops in user_ops.items() for op in ops]
    for userid, ops in user_ops.items():
        for op in op_categories:
            if op in ops:
                if op not in op_churn:
                    op_churn[op] = [0, 0]
                op_churn[op][0] += 1
        if userid == 262834177: 
            pass
        elif user_label[userid] == 1:               
            op_churn[ops[-1]][1] += 1 # 对最后一个动作进行处理
    return op_churn


def merge():    
    op_churn = churn_rate()
    op_intervals = count_opintervals()
    
    op_churnrate = {}
    
    ins = []
    rates = []
    for op, num in op_churn.items():
        op_churnrate[op] = num[1] * 1.0 / num[0] 
    

    op_in = {}
    for op, intervals in op_intervals.items():
        if len(intervals) >= 10:
            #print(len(intervals))
            intervals.remove(max(intervals))
            intervals.remove(max(intervals))
            intervals.remove(min(intervals))
            intervals.remove(min(intervals))
            #print(len(intervals))
        res = np.mean(intervals)
        op_in[op] = res
        ins.append(res)
        rates.append(op_churnrate[op])
       
    import pandas as pd
    s1 = pd.Series(ins)
    s2 = pd.Series(rates)
    corr = s1.corr(s2)
    print(corr) # 0.0905729035325 # 0.1046810
   
    from pyecharts import Line
    line = Line("")

    attr = [_ for _ in range(len(ins))]
    churnrates = [rate * 1000 for rate in rates]

    new_intervals = [i for i in ins[::10]]
    new_churnrates = [c for c in churnrates[::10]]
    attr = [i for i in range(0, len(ins), 10)]
    line = Line("")
    line.add("动作随后时间间隔", attr, new_intervals)
    line.add("动作留存比 * 1000", attr, new_churnrates)
    line.show_config()
    line.render()
    
if __name__ == '__main__':
    merge()
    pass
    



    
    

    



