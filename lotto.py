import numpy as np
import os
import collections

import lunar

class Lotto(object):
    
    def __init__(self, history=None, lotto_type=None):
        self.dataframe = history
        if history is not None:
            self.n_periods = history.shape[0]
        self.lotto_type = lotto_type
        self.nl_year_sx = lunar.Lunar().sx_year()
        self.zbfuncs = {
            'mod': self.mod, # 模
            'fen': self.fen, # 分
            'ls': self.ls, # 路数
            'kd': self.kd, # 跨度
            'ws': self.ws, # 尾数
            'hs': self.hs, # 和数
            'bs': self.bs, # 波色
            'sx': self.sx, # 生肖
            'wx': self.wx, # 五行
            'jl': self.jl, # 距离
            'yqc': self.yqc, # 与前差
            'pre': self.pre, # 在前期
            'hot': self.hot, # 冷热
            'rnd': self.rnd, # 随机
            'formula': self.formula, # 公式
        }
    
    def adapter(self, zb_func, ):
        pass
    
    
    def mod(self, column, m=2, mod_list=None):
        '''模'''
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        if mod_list is None:
            mod_list = [i for i in range(m) if i%2 != m%2]
        res = []
        for x in series:
            if x % m in mod_list:
                res.append(1)
            else:
                res.append(-1)
        return np.array(res)
        
        
    def fen(self, column, f=2, star=2, fen_list=None, lotto_type=None):
        '''分'''
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        if lotto_type is None:
            lotto_type = self.lotto_type
        
        if lotto_type == 'MS':
            max_number = 49
        else:
            max_number = 10 ** star - 1
        if fen_list is None:
            fen_list = [i for i in range(f) if i%2 != f%2]
        res = []
        for x in series:
            if x // ((max_number + 1) / f) in fen_list:
                res.append(1)
            else:
                res.append(-1)
        return np.array(res)
        
        
    def ls(self, column):
        '''路数'''
        # 单零路
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        res = []
        for x in series:
            shi = (x // 10) % 10
            ge = x % 10
            if (shi % 3) * (ge % 3) == 0 and (shi % 3) + (ge % 3) != 0:
                res.append(1)
            else:
                res.append(-1)
        return np.array(res)
        
        
    def kd(self, column):
        '''跨度'''
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        res = []
        for x in series:
            wei = [x // 10000, (x // 1000) % 10, (x // 100) % 10, (x // 10) % 10, x % 10]
            if 0 < max(wei) - min(wei) < 4:
                res.append(-1)
            else:
                res.append(1)
        return np.array(res)
        
        
    def ws(self, column, ws_type=None):
        '''尾数'''
        # 两种情况：尾数大小dx，尾数质合zh
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        if ws_type is None:
            ws_type = 'wd'
            
        if ws_type == 'wz':
            ws_list = [1,2,3,5,7]
        else:
            ws_list = [5,6,7,8,9]
            
        res = []
        for x in series:
            if x%10 in ws_list:
                res.append(1)
            else:
                res.append(-1)
        return np.array(res)
       
        
    def hs(self, column, star=2, hs_type=None, lotto_type=None):
        '''和数'''
        # 四种情况：和数大小dx，和数单双ds，和数尾数大小wsdx，和数尾数质合wszh
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        if (hs_type is None) or (not hs_type in ['hdx', 'hds', 'hwd', 'hwz']):
            hs_type = 'hdx'
        
        if lotto_type is None:
            lotto_type = self.lotto_type
            
        sum_series  = series % 10
        sum_series += (series // 10) % 10
        sum_series += (series // 100) % 10
        sum_series += (series // 1000) % 10
        sum_series += (series // 10000) % 10
            
        if hs_type == 'hdx':
            if lotto_type == 'MS':
                half = (4 + 9) / 2
            else:
                half = 9 * star / 2
            
            res = []
            for x in sum_series:
                if x > half:
                    res.append(1)
                else:
                    res.append(-1)
            return np.array(res)                    # 和数大小dx
        elif hs_type == 'hds':
            return self.mod(sum_series)             # 和数单双ds
        elif hs_type == 'hwd':
            return self.ws(sum_series)              # 和数尾数大小wsdx
        elif hs_type == 'hwz':
            return self.ws(sum_series, ws_type='wz')# 和数尾数质合wszh
        
        
    def bs(self, column, color='g'):
        '''波色'''
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        ix = 'rgb'.index(color)
        colors = [[1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46], # 红(17)
                  [5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49],   # 绿(16)
                  [3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48]]    # 篮(16)
        res = []
        for x in series:
            if x in colors[ix]:
                res.append(-1)
            else:
                res.append(1)
        return np.array(res)
        
        
    def sx(self, column, nl_year_sx=None):
        '''生肖'''
        # 家野
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        sx = '鼠牛虎兔龙蛇马羊猴鸡狗猪'
        jia_sx = '牛马羊鸡狗猪'
        
        if nl_year_sx is None:
            nl_year_sx = self.nl_year_sx
        
        ix = sx.index(nl_year_sx)
        if ix<11:
            tmp_sx = sx[ix+2:] + sx[:ix+2]
        else:
            tmp_sx = sx[1:] + sx[0]
        new_sx = list(tmp_sx)
        new_sx.reverse()
        
        res = []
        for x in series:
            if new_sx[x%12] in jia_sx:
                res.append(1)
            else:
                res.append(-1)
        return np.array(res)
    
    
    def wx(self, column):
        '''五行'''
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
            
        pass
    
    
    def jl(self, column, column2=None, pre=1, star=2, lotto_type=None):
        '''距离'''
        # 适用于相异两列。
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        if column2 is None:
            series2 = series
        else:
            if isinstance(column2, int):
                series2 = self.dataframe[:,column2-1]
            elif isinstance(column2, np.ndarray):
                series2 = column2
                
        if lotto_type is None:
            lotto_type = self.lotto_type
            
        if lotto_type == 'MS':
            max_number = 49
        else:
            max_number = 10 ** star - 1
            
        res = []
        for i in range(self.n_periods):
            if i < pre:
                res.append(-1)
                continue
            
            if max_number/4 < np.abs(series[i] - series2[i-pre]) < 3 * max_number/4:
                res.append(1)
            else:
                res.append(-1)
        return np.array(res)
    
    
    def yqc(self, column, column2=None, pre=1):
        '''与前差'''
        # 适用于相异两列。
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        if column2 is None:
            series2 = series
        else:
            if isinstance(column2, int):
                series2 = self.dataframe[:,column2-1]
            elif isinstance(column2, np.ndarray):
                series2 = column2
            
        res = []
        for i in range(self.n_periods):
            if i < pre:
                res.append(1)
                continue
            
            if series[i] > series2[i-pre]:
                res.append(1)
            elif series[i] < series2[i-pre]:
                res.append(-1)
            else:
                res.append(res[-1])
        return np.array(res)

    
    def pre(self, column, column2=None, pre=49):
        '''在前期'''
        # 适用于相异两列。series2=dataframe时，适用于多列。
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        if column2 is None:
            series2 = series
        elif column2 == 0:
            series2 = self.dataframe
        elif 1 <= column2 <= 7:
            series2 = self.dataframe[:,column2-1]
        elif isinstance(column2, np.ndarray):
            series2 = column2
            
        res = [-1]
        for i in range(1, self.n_periods):
            if i < pre:
                start = 0
            else:
                start = i - pre
            
            if series[i] in series2[start:i]:
                res.append(1)
            else:
                res.append(-1)
        return np.array(res)
    

    def hot(self, column, column2=None, star=2, jiaquan_type=None, lotto_type=None):
        '''冷热'''
        # 适用于相异两列。series2=dataframe时，适用于多列。
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        if column2 is None:
            series2 = series
        elif column2 == 0:
            series2 = self.dataframe
        elif 1 <= column2 <= 7:
            series2 = self.dataframe[:,column2-1]
        elif isinstance(column2, np.ndarray):
            series2 = column2
            
        if lotto_type is None:
            lotto_type = self.lotto_type
            
        if lotto_type == 'MS':
            max_number = 49
            min_number = 1
        else:
            max_number = 10 ** star - 1
            min_number = 0
            
        n_numbers = max_number - min_number + 1 # 号码个数
        s = sum(range(1, n_numbers + 1))
        res = []
        for i in range(self.n_periods):
            tmp = [0] * (n_numbers)
            for j in range(1, n_numbers + 1):
                if i - j < 0:
                    break
                    
                if jiaquan_type is None or not jiaquan_type in ['xx', 'tl', 'zs', 'js']:
                    jiaquan_type = 'xx'
                    
                if jiaquan_type == 'xx':
                    tmp[series2[i-j] - 1] += (n_numbers + 1 - j) / s # 线性加权
                elif jiaquan_type == 'tl':
                    tmp[series2[i-j] - 1] += 1 / (j + 1)             # 泰勒加权
                elif jiaquan_type == 'zs':
                    tmp[series2[i-j] - 1] += 1 / 2**j                # 指数加权
                else:
                    tmp[series2[i-j] - 1] += 1                       # 简单计数
            #c = collections.Counter(tmp)
            #hots = [x[0] for x in c.most_common(25)] # 找出最热的25个数
            #tmp = list(enumerate(tmp)) # 错误：从0开始
            tmp = list(zip(range(1,50),tmp))
            tmp.sort(key=lambda x:x[1], reverse=True)
            ttmp = [x[0] for x in tmp[:n_numbers//2]] # 取出最多的一半
            if series[i] in ttmp:
                res.append(1)
            else:
                res.append(-1)
                
        return np.array(res)
        
        
    def rnd(self, column, filename='rnd.txt', new=False, scale=0.5, star=2, lotto_type=None):
        '''随机'''
        # 若scale改变，则必须new=True
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        if lotto_type is None:
            lotto_type = self.lotto_type
            
        if lotto_type == 'MS':
            max_number = 49
            min_number = 1
        else:
            max_number = 10 ** star - 1
            min_number = 0
            
        n_numbers = max_number - min_number + 1 # 号码个数

        filename = os.path.join(os.path.dirname(__file__), filename)
        if not os.path.exists(filename) or new == True:
            tmp = np.arange(min_number, max_number)
            np.random.shuffle(tmp)
            np.savetxt(filename, tmp[:int(n_numbers*scale)])
        random_list = np.loadtxt(filename).astype(int)
        

        res = []
        for i in range(series.size):
            if series[i] in random_list:
                res.append(1)
            else:
                res.append(-1)
        return np.array(res)
    
    
    def formula(self, column, formula_text=''):
        '''公式'''
        if isinstance(column, int):
            series = self.dataframe[:,column-1]
        elif isinstance(column, np.ndarray):
            series = column
        
        pass
    
    
    def tongji(self, dataframe, func_name=None, to_column=7, pre=1):
        '''统计'''
        dataframe = self.dataframe
        
        funcs = {
                 'bs': self._bs, 
                 'sx': self._sx, 
                 'wx': self._wx,
        }
        if func_name is None:
            func_name = 'bs'
        if func_name == 'bs':
            values_list = range(3)
        elif func_name == 'sx':
            values_list = range(12)
        elif func_name == 'wx':
            values_list = range(5)
            
        if not to_column in  range(1, 8):
            to_column = 7
            
        res = [-1]
        for i in range(1, self.n_periods):
            tmp = [funcs[func_name](x) for x in dataframe[i-pre]]
            count_values = [tmp.count(j) for j in values_list]
            if funcs[func_name](dataframe[i, to_column]) == self._strage(count_values): # 按照策略strage
                res.append(1)
            else:
                res.append(-1)
        return np.array(res)
    
    
    def _strage(self, count_values):
        
        return count_values.index()
    
    
    def test(self):
        '''测试静态指标'''
        series = np.arange(1,50) #生成49个测试数据
        mf = np.arange(2,17) # 模（分）参数，用于循环测试模、分指标
        print('各指标区分情况：')
        for x in mf:
            print('M{}：{}'.format(x, np.sum(self.mod(series, m=x))))
        for x in mf:
            print('F{}：{}'.format(x, np.sum(self.fen(series, f=x, lotto_type='MS'))))
        print('WD：{}'.format(np.sum(self.ws(series, ws_type='wd')))) # 尾大
        print('WZ：{}'.format(np.sum(self.ws(series, ws_type='wz')))) # 尾质
        print('HDX：{}'.format(np.sum(self.hs(series, hs_type='hdx')))) # 和大
        print('HDS：{}'.format(np.sum(self.hs(series, hs_type='hds')))) # 和单
        print('HWD：{}'.format(np.sum(self.hs(series, hs_type='hwd')))) # 和尾大
        print('HWZ：{}'.format(np.sum(self.hs(series, hs_type='hwz')))) # 和尾质
        print('BS：{}'.format(np.sum(self.bs(series, color='g')))) # 波色
        print('SX：{}'.format(np.sum(self.sx(series))))        # 生肖
        print('RND：{}'.format(np.sum(self.rnd(series)))) # 随机
        print('LS：{}'.format(np.sum(self.ls(series))))   # 路数
        print('KD：{}'.format(np.sum(self.kd(series))))   # 
        print(self.kd(series))
        

    def make_forge_history(self, periods=500, lotto_type='MS'):
        '''生成伪造历史数据（六合）'''
        history = np.zeros((periods, 7), dtype='int')
        for i in range(periods):
            tmp = np.arange(1,50)
            np.random.shuffle(tmp)
            history[i] =  tmp[:7]
            
        self.dataframe = history
        self.n_periods = history.shape[0]
        self.lotto_type = lotto_type
        
    def make_real_history(self, periods=500, filename='history.csv', lotto_type=None):
        '''生成真实历史数据'''
        filename = os.path.join(os.path.dirname(__file__), filename)
        history = np.genfromtxt(filename, delimiter=',')
        
        self.dataframe = history[-periods:,1:-1].astype(int)
        self.n_periods = self.dataframe.shape[0]
        self.lotto_type = lotto_type
        
    def test2(self):
        '''测试动态指标'''
        self.make_history(periods=500)
        series = self.dataframe[:,6]
        self.jl(series, series2=None, pre=1, star=2, lotto_type='MS')
        
    
    
    # ===============================================================
    def get_numbers(self, zb_func, dictory=-1):
        '''根据指标函数及方向，选取号码'''
        cond = np.array(list(map(lambda z:True if zb_func(z)==1 else False, range(1,50))))
        x = np.where(cond, range(1,50), cond)
        res = x[x>0]
        return res
    # ===============================================================
    
    
def make_test_series(periods=1000):
    return np.random.randint(1, 50, periods)
    
    
def get_new_draw():
    url = 'http://www.5457.com/chajian/bmjg.js'
    import requests
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; rv:43.0) Gecko/20100101 Firefox/43.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,en-US;q=0.7,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate',
        #'If-Modified-Since': 'Tue, 19 Jan 2016 13:34:53 GMT'
    }

    r = requests.get(url, headers=headers)
    dic = r.json() # {"k":"008,39,25,30,08,20,18,13,009,01,21,四,21点30分","t":"1000","联系":"QQ：7136995"}
    haoma = dic['k'].split(',')[1:8] # ['39','25','30','08','20','18','13']
    haoma_numbers = [int(x) for x in haoma]
    sx = shengxiao = [get_sx_text(x) for x in haoma_numbers]
    return dic['k'][:24] + ',' + ''.join(shengxiao)
    
def get_sx_text(num, sx_year='yang'): 
    '''取号码的生肖名'''
    sx_years = ['shu', 'niu', 'hu', 'tu', 'long', 'she', 'ma', 'yang', 'hou', 'ji', 'gou', 'zhu']
    ix = sx_years.index(sx_year) # 得到年生肖的索引
    
    sx = '鼠牛虎兔龙蛇马羊猴鸡狗猪'
    return sx[ix - (num%12 - 1)]
    
def get_jy_text(num, sx_year='yang'): 
    '''取号码的家野名'''
    sx = get_sx_text(num, sx_year=sx_year)
    jiaye_sx = ['牛马羊鸡狗猪','鼠虎兔龙蛇猴']
    jiaye_name = '家野'
    for i,sxs in enumerate(jiaye_sx):
        if sx in sxs:
            return jiaye_name[i]
    
def get_bs_text(num):
    '''取号码的波色名'''
    colors_names = '红绿蓝'
    colors_numbers = [[1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46], # 红(17)
                      [5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49],   # 绿(16)
                      [3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48]]    # 篮(16)
    for i, c in enumerate(colors_numbers):
        if num in c:
            return colors_names[i]

def get_ds_text(num):
    '''取号码的单双名'''
    ds = '双单'
    return ds[num%2]

if __name__ == '__main__':
    periods=1000
    begin_number = 1
    end_number = 49
    history = np.random.randint(begin_number, end_number + 1, periods)
    
    
    #print(history)
    #lt = Lotto(history)
    #lt.test()
    #s = lt.mod(history)
    #print(sum(s))
    #s = lt.fen(history, lotto_type='MS')
    #print(sum(s))
    #s = lt.ws(history)
    #print(sum(s))
    #s = lt.hs(history)
    #print(sum(s))
    
    #for i in range(49):
    #    print('{} --> {} {} {} {}'.format(i+1, get_sx_text(i+1, sx_year='yang'), get_bs_text(i+1), get_ds_text(i+1), get_jy_text(i+1)))
    
    # 取最新开奖结果
    print(get_new_draw())

    