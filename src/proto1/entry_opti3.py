import geatpy as ea
import numpy as np
from ode_solver_varh import pre_eval_ode_functions
import matplotlib.pyplot as plt
import multiprocessing
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 优化目标个数
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 7  # 初始化Dim（决策变量维数）
        Dim = 3  # 初始化Dim（决策变量维数）
        # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        varTypes = [0] * Dim

        # 设置决策变量的上下界，除最后一个外都是-10到10，最后一个是-10到-1
        lb =[0]+ [-2] * (Dim-1) 
        #ub = [10] * (Dim - 1) + [-1]  # 最后一个上界是-1
        ub = [2]+[2] * (Dim-1) 
        lbin = [1] * Dim  # 所有变量的下界都包含
        ubin = [1] * Dim  # 所有变量的上界都包含
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)


    def evalVars(self, Vars):  # 目标函数
        # f11 = Vars ** 2
        # f21 = (Vars - 2) ** 2
        # ObjV1 = np.hstack([f11, f21])  # 计算目标函数值矩阵
        # CV1 = -Vars ** 2 + 2.5 * Vars - 1.5  # 构建违反约束程度矩阵
        print(Vars)
        results=pre_eval_ode_functions(Vars)
        f1=[]
        f2=[]
        CV=[]
        for i, (case_key, case_data) in enumerate(results.items(), start=1):
            if isinstance(case_data, str):  # Check if the result is an error message (string)
                print(f"{case_key} encountered an error: {case_data}")
                f1.append([999999999])
                f2.append([10000])
                CV.append([1])
                continue  # Skip to the next iteration
            
            # Proceed with unpacking and plotting since case_data is not an error string
            times, ys, true_ys, errors = case_data["times"], case_data["ys"], case_data["true_ys"], case_data["errors"]
            total_function_calls, rmse,max_diff = case_data["total_function_calls"], case_data["rmse"],case_data["max_diff"]
            #plt.subplot(len(results), 2, 2*i-1)
            # plt.subplot(2, 1, 1)
            # plt.plot(times, ys, 'r-', label='RK45 Approximation')
            # plt.plot(times, true_ys, 'b--', label='True Solution $y=\sin(t)$')
            # plt.xlabel('Time (t)')
            # plt.ylabel('Solution (y)')
            # plt.title(f'{case_key}-Total f calls: {total_function_calls}')
            # plt.legend()
            
            # # Plotting the error for the current case
            # #plt.subplot(len(results), 2, 2*i)
            # plt.subplot(2, 1, 2)
            # plt.plot(times, errors, 'g-', label='Error')
            # plt.xlabel('Time (t)')
            # plt.ylabel('Error')
            # plt.yscale('log')
            # plt.title(f'Error vs. Time for {case_key}-max_diff: {max_diff}')
            # plt.legend()
            # plt.savefig(f'plot_output_{case_key}.png', dpi=300)
            # plt.close()

            f1.append([total_function_calls])
            f2.append([np.log10(max_diff)])
            CV.append([0])

            
        ObjV = np.hstack([f1,f2])
        print(ObjV)
        return ObjV, np.array(CV)

def main():
# 实例化问题对象

    start_population = np.array([
    # [1, 0.5,1.0],
    [0.5,0.5,1.0],
    [1.909715817293929918, 7.378215165329875536e-01, 1.326320112456623179e-01],
    # [9.885160759978742773e-01, 0.5443494443274259353,-4.817238145637867675e-01],
#     [1.989807128906250000e+00,1.143527049592940187e-01,1.158284883010818334e+00],
# [1.768560950218853456e-01,3.244628906250000000e-01,-4.158935546875000000e-01],
# [1.698980914024604605e+00,6.740714887772683017e-01,9.940885557609253631e-01],
# [1.610595703125000000e+00,9.587228997988022083e-01,1.426282987627170540e+00],
# [1.501778262184938706e+00,2.566056074435958401e-01,7.578116955080971273e-01],
# [1.400183459350502924e+00,1.711716453219317646e+00,1.517446346180373329e+00],
# [1.793334960937500000e+00,6.902142413088354633e-01,1.767927701131213336e+00],
# [1.989807128906250000e+00,1.143527049592940187e-01,1.387343319927693663e+00],
# [5.687607400421733628e-01,-1.200202754726496401e+00,1.506795494032051153e+00],
# [1.498291261139789388e+00,1.972782349753335662e+00,1.362531169619120242e+00],
# [1.409677085792629070e+00,6.979770091101507035e-01,1.392387835144781594e+00],
# [1.690555662695130357e+00,1.429467394724919793e-01,-6.667952254556532843e-01],
# [8.967084803429485607e-02,-2.739257812500000000e-01,1.383332123452404971e+00],
# [1.847234535527465082e+00,-2.064457307105784512e-01,-1.446228928812956394e+00],
# [1.798294983081958121e+00,-1.124896457300164909e-01,-1.446228928812956394e+00],
# [5.185546875000000000e-01,-1.416015625000000000e-02,-7.519531250000000000e-01],
# [1.610595703125000000e+00,4.462890625000000000e-01,-1.431762695312500000e+00],
# [1.719620299478818559e+00,-1.199707031250000000e+00,1.775807031032653693e+00],
# [1.172363281250000000e+00,-1.221557617187500000e+00,1.186889648437500000e+00],
# [1.793334960937500000e+00,1.101508612280677424e+00,1.392387835144781594e+00],

    ], dtype=np.float64)

    ode_problem = MyProblem()
    # Desired population size
    pop_size = 200
    # Option 1: Duplicate the two individuals to fill the population
    # Repeat start_population to fill up to pop_size
    #duplicated_population = np.tile(start_population, (pop_size // len(start_population) + 1, 1))[:pop_size]

    population = ea.Population(Encoding='RI', NIND=pop_size)


    # 构建算法
    algorithm = ea.moea_NSGA2_templet(ode_problem,
                                        population,
                                        MAXGEN=100,  # 最大进化代数
                                        logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。

    # 求解
    res = ea.optimize(algorithm,prophet=start_population, seed=1, verbose=False, drawing=1, outputMsg=True, drawLog=False, saveFlag=True, dirName='202411111510')





if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()