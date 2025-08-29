import random

import config

city_dist_mat = None
config = config.get_config()

# 参数
gene_len = config.city_num
individual_num = config.individual_num
gen_len = config.gen_num  # 迭代次数
mutate_prob = config.mutate_prob


def copy_list(old_arr):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


# 个体类
class Individual:
    def __init__(self, genes=None):  # 这里要区分是初代种群还是交叉变异得到的种群
        if genes is None:
            genes = [i for i in range(gene_len)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        fitness = 0.0
        for i in range(gene_len - 1):
            # 起始和目标城市
            from_idx, to_idx = self.genes[i], self.genes[i + 1]
            fitness += city_dist_mat[from_idx, to_idx]
        # 连接首尾，构成回路
        fitness+=city_dist_mat[self.genes[-1],self.genes[0]]
        return fitness

class Ga:
    def __init__(self,inputs):
        global city_dist_mat
        city_dist_mat=inputs
        self.best=None  # 每一代的最优解（核心）
        self.individual_list=[]
        self.result_list=[]
        self.fitness_list=[]

    def cross(self):
        new_gen=[]  # 交叉产生的新个体列表
        random.shuffle(self.individual_list)
        for i in range(0,individual_num-1,2):
            # 一次挑选两个父亲基因便于交叉互换
            genes1=copy_list(self.individual_list[i].genes)
            genes2=copy_list(self.individual_list[i+1].genes)
            index1=random.randint(0,gene_len-2)
            index2=random.randint(index1,gene_len-1)
            pos1_recorder={value:idx for idx,value in enumerate(genes1)}
            pos2_recorder={value:idx for idx,value in enumerate(genes2)}
            for j in range(index1,index2):
                value1,value2=genes1[j],genes2[j]
                pos1,pos2=pos1_recorder[value2],pos2_recorder[value1]
                genes1[j],genes1[pos1]=genes1[pos1],genes1[j]
                pos1_recorder[value1],pos1_recorder[value2]=pos1,j
                pos2_recorder[value2],pos2_recorder[value1]=j,pos2
            new_gen.append(Individual(genes1))
            new_gen.append(Individual(genes2))
        return new_gen

    def mutate(self,new_gen):
        for individual in new_gen:
            if random.random()<mutate_prob:
                # 变异机制：翻转基因内某一段序列
                old_genes=copy_list(individual.genes)
                index1=random.randint(0,gene_len-2)
                index2=random.randint(index1,gene_len-1)
                genes_mutate=old_genes[index1:index2]
                genes_mutate.reverse()
                individual.genes=old_genes[:index1]+genes_mutate+old_genes[index2:]
        self.individual_list+=new_gen

    def select(self):
        # 锦标赛选择机制
        # 分组查优--防止陷入局部最优
        group_num=10 # 小组数
        group_size=10 # 单个小组人数
        group_winner=individual_num//group_num  # 每小组获胜人数
        winners=[]
        for i in range(group_num):
            group=[]
            for j in range(group_size):
                player=random.choice(self.individual_list)
                player=Individual(player.genes)
                group.append(player)
            group=Ga.rank(group)
            winners+=group[:group_winner]
        self.individual_list=winners

    @staticmethod
    def rank(group):
        # 选择排序
        for i in range(len(group)-1):
            for j in range(i+1,len(group)):
                if group[i].fitness>group[j].fitness:
                    group[j],group[i]=group[i],group[j]
        return group

    def next_gen(self):
        # 交叉->变异->选择
        new_gen=self.cross()
        self.mutate(new_gen)
        self.select()
        for individual in self.individual_list:
            if individual.fitness<self.best.fitness:
                self.best=individual

    def train(self):
        # 初代种群
        self.individual_list=[Individual() for _ in range(individual_num)]
        self.best=self.individual_list[0]
        for i in range(gen_len):
            self.next_gen()
            result=copy_list(self.best.genes)
            result.append(result[0])  # 首尾相接
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        return self.result_list,self.fitness_list
