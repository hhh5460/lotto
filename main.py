import threading
import time

import numpy as np

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import lotto
import kline

import bpnn

# 计时（装饰器）
def time_it(func):
    def wrap(*args):
        print('开始计时。。。')
        t0 = time.time()
        func(*args)
        t1 = time.time()-t0
        print('耗时: {}分{}秒'.format(t1//60, int(t1)%60))
        
    return wrap


# 线程（装饰器）
def thread_it(func):
    def wrap(*args):
        print('开始线程。。。')
        t = threading.Thread(target=func, args=args) 
        t.setDaemon(True)   # 守护--就算主界面关闭，线程也会留守后台运行（不对!）
        t.start()           # 启动
        # t.join()          # 阻塞--会卡死界面！
        
    return wrap
    
    
    
class Application(tk.Tk):
    '''程序'''
    def __init__(self):
        '''初始化'''
        super().__init__() # 有点相当于tk.Tk()
        self.wm_title("Tk 绑定 matplotlib")
        self.iconbitmap(default="Coffee.ico")
        
        self.setupUI() # 生成界面
        self.draw()    # 绘图逻辑
        
        self.classifiers = {'bpnn': bpnn.BPNN} # 预测用的分类器
        
        
    def setupUI(self):
        '''生成界面'''
        # 上部绘图区域
        fig = Figure(figsize=(6,4), dpi=100)
        #self.ax = fig.add_subplot(111)
        self.ax1 = fig.add_axes([0, 0.2, 1, 0.8]) # 上图位置、大小
        #self.ax1.yaxis.grid()
        self.ax1.grid()
        self.ax1.set_xticks([])
        self.ax2 = fig.add_axes([0, 0, 1, 0.2]) # 下图位置、大小
        self.ax2.yaxis.grid()
        self.ax2.axhline()
        self.ax2.set_axis_bgcolor('#AADDFF')
        #self.ax = fig.add_axes([0, 0, 1, 1])
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Hide the right and top spines
        #self.ax1.spines['right'].set_visible(False)
        #self.ax1.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        #self.ax1.yaxis.set_ticks_position('left')
        #self.ax1.xaxis.set_ticks_position('bottom')
        
        #print(self.ax1.spines)

        
        # 下部面板
        footframe = tk.Frame(master=self)
        footframe.pack(side=tk.LEFT)
        
        kernel_group = tk.Frame(footframe)
        
        self.lotto_type = tk.StringVar()
        combobox1 = ttk.Combobox(kernel_group, values=("香港","重庆","天津","新疆"), textvariable=self.lotto_type, state='readonly')
        combobox1.current(0)
        combobox1.bind("<<ComboboxSelected>>", self.selectCombobox)
        tk.Label(kernel_group, text="彩票类型", anchor="e", width=7).grid(row=0, column=0)
        combobox1.grid(row=0, column=1)
        
        self.column_index = tk.IntVar()
        combobox2 = ttk.Combobox(kernel_group, values=(1,2,3,4,5,6,7), textvariable=self.column_index, state='readonly')
        combobox2.current(0)
        combobox2.bind("<<ComboboxSelected>>", self.selectCombobox)
        tk.Label(kernel_group, text="列", anchor="e", width=7).grid(row=1, column=0)
        combobox2.grid(row=1, column=1)
        
        self.zhibiao_type = tk.StringVar()
        combobox3 = ttk.Combobox(kernel_group, values=("M2","M3","M4","M5","M6","M7","M8","M10","M12"), textvariable=self.zhibiao_type, state='readonly')
        combobox3.current(0)
        combobox3.bind("<<ComboboxSelected>>", self.selectCombobox)
        tk.Label(kernel_group, text="指标", anchor="e", width=7).grid(row=2, column=0)
        combobox3.grid(row=2, column=1)
        
        self.ml_type = tk.StringVar()
        combobox4 = ttk.Combobox(kernel_group, values=("bpnn","svm","random forest","vote"), textvariable=self.ml_type, state='readonly')
        combobox4.current(0)
        combobox4.bind("<<ComboboxSelected>>", self.selectCombobox)
        tk.Label(kernel_group, text="预测", anchor="e", width=7).grid(row=3, column=0)
        combobox4.grid(row=3, column=1)
        kernel_group.pack(side=tk.LEFT)

        btn0 = ttk.Button(kernel_group, text='散点图', command=lambda :self.scatter())
        btn0.grid(row=1, column=2)
        
        btn1 = ttk.Button(kernel_group, text='画图', command=lambda :self.draw())
        btn1.grid(row=2, column=2)
        #btn1.bind('<Return>', lambda e:thread_it(self.draw()))
        
        btn2 = ttk.Button(kernel_group, text='预测', command=lambda :self.yuce_test2(100, 10))
        btn2.grid(row=3, column=2)
        #btn2.bind('<Return>', lambda e:thread_it(self.yuce))
        
    def scatter(self):
        '''散点图'''
        n_point = 200
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        X, y = self.dataframe[:-1], self.series[1:]
        xs, ys, zs = X[-n_point:,7], X[-n_point:,8], X[-n_point:,10]
        ax.scatter(xs, ys, zs, c=np.where(y[-n_point:]==-1,'r','b'))
        
        plt.show()
        
    #@thread_it
    #@time_it
    def draw(self):
        '''绘图逻辑'''
        lt = lotto.Lotto()
        lt.make_real_history(periods=1000, lotto_type='MS') # 生成真实历史数据
        
        series = lt.rnd(7, filename='rnd.txt', new=True, scale=0.5, star=2, lotto_type='MS')
        kl = kline.KLine(series)
        
        kl.plot2(ax1=self.ax1, ax2=self.ax2, period=200, n_series=12)
        
        #self.ax.set_title('演示：画N个随机点', fontproperties="FangSong")
        self.canvas.show()
        
        self.dataframe = kl.dataframe # 用于预测
        self.series = series
        
    @thread_it
    @time_it
    def yuce(self):
        '''预测'''
        if self.dataframe is None: raise 'self.dataframe is None!'
        X, y = self.dataframe[:-1], self.series[1:]
        print(y)
        x = self.dataframe[-1]
        clf = bpnn.BPNN([12,15,2])
        clf.fit(X[-50:], y[-50:], print_loss=True) # nn_hdim=8
        return clf.predict(x)
        
    @thread_it
    @time_it
    def yuce_test(self, n=100):
        '''预测100期，看看效果'''
        if self.dataframe is None: raise 'self.dataframe is None!'
        X, y = self.dataframe[:-1], self.series[1:]
        
        res = []
        clf = bpnn.BPNN([12,15,2]) 
        for i in range(n,-1,-1):
            x_test, y_test = X[-50-i+1], y[-50-i+1] # 用前50期数据预测
            clf.fit(X[-50-i:,7:], y[-50-i:], print_loss=False) # nn_hdim=8
            res.append((clf.predict(x_test[7:]) == y_test)[0])
            
        print(np.sum(res)/len(res))
        print(res)

    @thread_it
    @time_it
    def yuce_test2(self, n_period=100, n_zb=10, n_pre=50):
        '''预测100期，看看效果'''
        # 10个指标，为方便起见，这里暂时用10次随机指标
        res = []
        for i in range(n_zb):
            self.draw()
            
            X, y = self.dataframe[:-1], self.series[1:]
            
            tmp = []
            clf = bpnn.BPNN([12,15,2]) 
            for i in range(n_period,-1,-1):
                x_test, y_test = X[-n_pre-i+1], y[-n_pre-i+1] # 用前50期数据预测
                clf.fit(X[-n_pre-i:], y[-n_pre-i:], print_loss=False) # nn_hdim=8
                tmp.append((clf.predict(x_test) == y_test)[0])
                
            res.append(tmp)
        r = np.sum(res, axis=0)
        print(r)
        rr = r[r>3]
        print(rr[rr<7].shape[0])
        
    def selectCombobox(self, event):
        print('event.type', event.type)
        print('event.state', event.state)
        print('event.widget', event.widget)

if __name__ == '__main__':
    # 实例化Application
    app = Application()
    
    # 主消息循环:
    app.mainloop()