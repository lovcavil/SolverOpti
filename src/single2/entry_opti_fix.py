import geatpy as ea
import numpy as np
from ode_solver import pre_eval_ode_functions
from ode_solver import pre_eval_ode_functions_fix
import matplotlib.pyplot as plt
import multiprocessing
import math
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 优化目标个数
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 3  # 初始化Dim（决策变量维数）

        # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        varTypes = [0] * (Dim-1)+ [1]
        # 设置决策变量的上下界，除最后一个外都是-10到10，最后一个是-10到-1
        # lb = [-1] * (Dim - 1) + [-5]
        # ub = [1] * (Dim - 1) + [-2]  
        lb = [0,0] + [-3]
        ub = [1,1] + [-2]  
        lbin = [1] * Dim  # 所有变量的下界都包含
        ubin = [1] * Dim  # 所有变量的上界都包含
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)


    def evalVars(self, Vars):  # 目标函数
        # f11 = Vars ** 2
        # f21 = (Vars - 2) ** 2
        # ObjV1 = np.hstack([f11, f21])  # 计算目标函数值矩阵
        # CV1 = -Vars ** 2 + 2.5 * Vars - 1.5  # 构建违反约束程度矩阵
        results=pre_eval_ode_functions_fix(Vars)
        f1=[]
        f2=[]
        CV=[]
        for i, (case_key, case_data) in enumerate(results.items(), start=1):
            if isinstance(case_data, str):  # Check if the result is an error message (string)
                print(f"{case_key} encountered an error: {case_data}")
                f1.append([999999999])
                f2.append([10])
                CV.append([1])
                continue  # Skip to the next iteration
            if case_data["issue"]:  # Check if the result is an error message (string)
                print(f"{case_key} encountered an issue")
                f1.append([999999999])
                f2.append([10])
                CV.append([1])
                continue  # Skip to the next iteration
            # Proceed with unpacking and plotting since case_data is not an error string
            times, ys, true_ys= case_data["times"], case_data["ys"], case_data["true_ys"]
            # errors = case_data["errors"]
            total_function_calls, rmse = case_data["total_function_calls"], case_data["rmse"]
            #plt.subplot(len(results), 2, 2*i-1)
            # plt.subplot(1, 2, 1)
            # plt.plot(times, ys, 'r-', label='RK45 Approximation')
            # plt.plot(times, true_ys, 'b--', label='True Solution $y=\sin(t)$')
            # plt.xlabel('Time (t)')
            # plt.ylabel('Solution (y)')
            # plt.title(f'{case_key}-Total f calls: {total_function_calls}')
            # plt.legend()
            
            # Plotting the error for the current case
            #plt.subplot(len(results), 2, 2*i)
            # plt.subplot(1, 2, 2)
            # plt.plot(times, errors, 'g-', label='Error')
            # plt.xlabel('Time (t)')
            # plt.ylabel('Error')
            # plt.yscale('log')
            # plt.title(f'Error vs. Time for {case_key}-RMSE: {rmse}')
            # plt.legend()
            # plt.savefig(f'plot_output_{case_key}.png', dpi=300)
            
            f1.append([math.log(total_function_calls)])
            f2.append([rmse])
            CV.append([0])

            plt.close()    
            
        ObjV = np.hstack([f1,f2])
        print(ObjV)
        return ObjV, np.array(CV)
def main():
# 实例化问题对象
    ode_problem = MyProblem()
    # 构建算法
    algorithm0 = ea.moea_NSGA2_templet(ode_problem,
                                        ea.Population(Encoding='RI', NIND=50),
                                        MAXGEN=200,  # 最大进化代数
                                        logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    algorithm = ea.moea_NSGA2_templet(ode_problem,
                                        ea.Population(Encoding='RI', NIND=50),
                                        MAXGEN=50,  # 最大进化代数
                                        logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # 求解
    res = ea.optimize(algorithm, seed=1, verbose=False, drawing=1, outputMsg=True, drawLog=False, saveFlag=True, dirName='result')
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()