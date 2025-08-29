import config
import numpy as np
import matplotlib.pyplot as plt
from ga import Ga

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

config = config.get_config()


def build_dist_mat(inputs):
    n = config.city_num
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = inputs[i, :] - inputs[j, :]
            # 计算距离的平方
            dist_mat[i, j] = np.dot(dist, dist)
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat


# 城市坐标
city_pos_list = np.random.rand(config.city_num, config.pos_dimension)
# 城市距离矩阵
city_dist_mat = build_dist_mat(city_pos_list)

print(city_pos_list)
print(city_dist_mat)

ga = Ga(city_dist_mat)
result_list, fitness_list = ga.train()
result = result_list[-1]
result_pos_list = city_pos_list[result, :]

# 可视化
# 路线图
fig1 = plt.figure()
plt.plot(result_pos_list[:, 0], result_pos_list[:, 1])
plt.scatter(
    city_pos_list[:, 0],  # 所有城市的x坐标
    city_pos_list[:, 1],  # 所有城市的y坐标
    c='blue',             # 点的颜色（蓝色，与路线红色区分）
    s=80,                 # 点的大小（数值越大越明显）
    marker='o',           # 点的形状（圆圈）
    label='城市坐标'       # 标签，用于图例
)
plt.title('路线')
plt.legend()
plt.grid()
plt.show()

# 适应度
fig2 = plt.figure()
plt.plot(fitness_list)
plt.title('适应度曲线')
plt.grid()
plt.show()
