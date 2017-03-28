# TextSimilarity
这是一个类，里面包含的有关文本相似度的常用的计算算法，例如，最长公共子序列，最短标记距离，TF-IDF等算法
例如简单简单简单的用法：创建类实例，参数是两个文件目录，之后会生成两个字符串a.str_a, a.str_b

a = TextSimilarity('/home/a.txt','/home/b.txt')
# In[89]:
a.minimumEditDistance(a.str_a,a.str_b)
Out[89]: 0.3273657289002558

# In[90]:
a.JaccardSim(a.str_a,a.str_b)
Out[90]: 0.17937219730941703

# In[91]: 
a.splitWordSimlaryty(a.str_a,a.str_b,sim=a.pers_sim)
Out[91]: 0.54331148827606712
