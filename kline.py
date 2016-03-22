import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import lotto

class KLine(object):
    
    def __init__(self, series):
        self.n_periods = series.size
        self.series = series # 从外接收的指标序列，形如：[1, -1, -1, -1, 1, -1, 1, 1 ,...]
        self.dataframe = np.zeros((self.n_periods, 12)) # 放置构建好的均线、布林线等序列
        
        self._build()
        
    def sum(self, sum_column = 0):
        '''走势线'''
        self.dataframe[0, sum_column] = self.series[0]
        for i in range(1, self.n_periods):
            self.dataframe[i, sum_column] = self.dataframe[i-1, sum_column] + self.series[i]
        
    def average(self, average_column=1, period=20, sum_column=0):
        '''均线'''
        for i in range(self.n_periods):
            if i < period:
                start = 0
                end = i
            else:
                start = i - period + 1
                end = i
            self.dataframe[i, average_column] = np.mean(self.dataframe[start:end+1, sum_column])
        
    def bolling(self, bolling_column=3, sum_column=0, average_column=1, period=20, beishu=2.0):
        '''布林通道'''
        for i in range(self.n_periods):
            if i < period:
                start = 0
                end = i
            else:
                start = i - period + 1
                end = i
            self.dataframe[i, bolling_column] = self.dataframe[i, average_column] + np.std(self.dataframe[start:end+1, sum_column]) * beishu
    
    def r_sum(self, rsum_column=7, sum_column=0, average_column=1):
        '''相对走势'''
        self.dataframe[:,rsum_column] = self.dataframe[:,sum_column] - self.dataframe[:,average_column]
        
    def trend(self, trend_column=8, average_column=1):
        '''趋势'''
        self.dataframe[0, trend_column] = self.dataframe[0, average_column]
        for i in range(1, self.n_periods):
            if self.dataframe[i, average_column] - self.dataframe[i-1, average_column] == 0:
                self.dataframe[i, trend_column] = self.dataframe[i-1, trend_column]
            else:
                self.dataframe[i, trend_column] = np.sign(self.dataframe[i, average_column] - self.dataframe[i-1, average_column])
    
    def width_bolling(self, width_bolling_column=10, average_column=1, bolling_column=3, xishu=1.0):
        '''布林通道宽度'''
        self.dataframe[:,width_bolling_column] = (self.dataframe[:,bolling_column] - self.dataframe[:,average_column]) * xishu
    
    def slope(self, slope_column=9, bolling_column=3, xishu=3.0):
        '''两条布林线切线倾斜角之差11'''
        self.dataframe[0, slope_column] = 0.0
        
        s_bolling1 = np.arctan(self.dataframe[1:, bolling_column] - self.dataframe[:-1, bolling_column])
        s_bolling2 = np.arctan(self.dataframe[1:, bolling_column+1] - self.dataframe[:-1, bolling_column+1])
        self.dataframe[1:, slope_column] = np.tan(s_bolling1 - s_bolling2) * xishu
    
    def _build(self):
        '''构建各个序列'''
        self.sum(sum_column=0)
        self.average()
        self.average(average_column=2, period=10)
        self.bolling()
        self.bolling(bolling_column=4, beishu=-2.0)
        self.bolling(bolling_column=5, beishu=1.0)
        self.bolling(bolling_column=6, beishu=-1.0)
        self.r_sum()
        self.trend(trend_column=8)
        self.slope()
        self.width_bolling()
        self.width_bolling(width_bolling_column=11, xishu=-1.0)
        
    def plot(self, period=100, n_series=12):
        '''作图'''
        plt.figure(dpi=80) #facecolor='white', edgecolor='#F3F9F1')
        
        if n_series < 7:
            ax1 = plt.axes([0.10, 0.05, 0.80, 0.90])  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
            ax1.grid()
        else:
            ax1 = plt.axes([0.10, 0.25, 0.80, 0.73]) # 上图位置、大小
            #ax1.yaxis.grid()
            ax1.grid()
            ax1.set_xticks([])
            ax2 = plt.axes([0.10, 0.02, 0.80, 0.20]) # 下图位置、大小
            ax2.yaxis.grid()
            ax2.axhline()
            ax2.set_axis_bgcolor('#AADDFF') # 
        ax1.set_axis_bgcolor('#AADDFF') # '#F3F9F1'
        #plt.style.use('ggplot')
        
        #markers = ['.',',','o','v','^','<','>','1','2','3','4', 's','p','*','h','H','+','x','D','d','|','_', r'$\clubsuit$']
        markers = []
        colors = ['#003366', '#FF0000', '#00FF00','#FFFF33','#FFFF33','#FFFF99','#FFFF99','#660000','#000033','#9999CC','#FF0099','#FF0099']
        linewidths = [2, 1.5, 1.0, 1.5, 1.5, 0.5, 0.5, 0.6, 2.0, 1.0, 0.5, 0.5]
        for i in range(n_series):
            if i == 0:
                ax1.plot(self.dataframe[-period:, i], marker='D', markersize = 4, linewidth=linewidths[i], color=colors[i]) #, linestyle=linestyle, color=color, linewidth=3)
                continue
            if i == 7:
                ax2.plot(self.dataframe[-period:, i], marker='x', markersize = 5, linewidth=linewidths[i],color=colors[i])
                continue
            if i < 7:
                ax1.plot(self.dataframe[-period:, i], linewidth=linewidths[i],color=colors[i]) 
            else:
                ax2.plot(self.dataframe[-period:, i], linewidth=linewidths[i],color=colors[i])
        
        plt.show()
        
    def plot2(self, ax1=None, ax2=None, period=100, n_series=12):
        '''作图'''
        ax1.clear()
        ax2.clear()
        
        ax1.set_axis_bgcolor('#AADDFF') # '#F3F9F1'
        #plt.style.use('ggplot')
        
        #markers = ['.',',','o','v','^','<','>','1','2','3','4', 's','p','*','h','H','+','x','D','d','|','_', r'$\clubsuit$']
        markers = []
        colors = ['#003366', '#FF0000', '#00FF00','#FFFF33','#FFFF33','#FFFF99','#FFFF99','#660000','#000033','#9999CC','#FF0099','#FF0099']
        linewidths = [2, 1.5, 1.0, 1.5, 1.5, 0.5, 0.5, 0.6, 2.0, 1.0, 0.5, 0.5]
        for i in range(n_series):
            if i == 0:
                ax1.plot(self.dataframe[-period:, i], marker='D', markersize = 4, linewidth=linewidths[i], color=colors[i]) #, linestyle=linestyle, color=color, linewidth=3)
                continue
            if i == 7:
                ax2.plot(self.dataframe[-period:, i], marker='x', markersize = 5, linewidth=linewidths[i],color=colors[i])
                continue
            if i < 7:
                ax1.plot(self.dataframe[-period:, i], linewidth=linewidths[i],color=colors[i]) 
            else:
                ax2.plot(self.dataframe[-period:, i], linewidth=linewidths[i],color=colors[i])
        
        #plt.show()
        
    def save_to_txt(self, filename):
        '''保存到文本'''
        filename = os.path.join(os.path.dirname(__file__), filename)
        np.savetxt(filename, self.dataframe, delimiter=',', newline='\r\n')
    
    
    
def make_test_series(period=1000):
    '''随机漫步，生成1000个数据'''
    a = np.random.randint(0, 2, period) # 0~1
    a = 2 * a - 1 # -1~1
    return a
    
if __name__ == '__main__':
    #series = make_test_series(500)
    
    lt = lotto.Lotto()
    #lt.make_forge_history(periods=1000, lotto_type='MS') # 生成伪造的历史数据
    lt.make_real_history(periods=1000, lotto_type='MS') # 生成真实历史数据
    
    #series = lt.mod(7, m=2) # 7表示第七个位置：特码位
    #series = lt.mod(7, m=3)
    #series = lt.mod(7, m=4)
    #series = lt.mod(7, m=16)
    #series = lt.fen(7, f=2, lotto_type='MS')
    #series = lt.fen(7, f=3, lotto_type='MS')
    #series = lt.fen(7, f=4, lotto_type='MS')
    #series = lt.fen(7, f=6, lotto_type='MS')
    #series = lt.fen(7, f=8, lotto_type='MS')
    #series = lt.fen(7, f=12, lotto_type='MS')
    #series = lt.ws(7,ws_type='wz')
    #series = lt.hs(7, hs_type='hdx', lotto_type='MS')
    #series = lt.hs(7,hs_type='hds')
    #series = lt.hs(7,hs_type='hwd')
    #series = lt.hs(7,hs_type='hwz')
    #series = lt.kd(7)
    #series = lt.ls(7)
    #series = lt.bs(7)
    #series = lt.sx(7)
    #series = lt.jl(7, column2=6, pre=1, star=2, lotto_type='MS')
    #series = lt.hot(7, column2=None, star=2, jiaquan_type='js', lotto_type='MS')
    #series = lt.pre(7, column2=None, pre=49)
    #series = lt.yqc(7, column2=None, pre=1)
    series = lt.rnd(7, filename='rnd.txt', new=True, scale=0.5, star=2, lotto_type='MS')
    
    kl = KLine(series)
    kl.plot(period=200, n_series=12)
    