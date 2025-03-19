# Project1   
# Deadline: 2025/03/21  23:59

Name: 

Student ID: 

The Project1 consists of two parts.

## PartA (30 pts)
There are 2 problems.
Please print it out and complete the question on it, and submit the paper version of the question to TA.

Every problem is 15 pts.

## PartB (70 pts)
**Create a virtual environment using [Anaconda](https://www.anaconda.com/download), with Python 3.6.13 and gym 0.9.4:**
```bash
conda create -n gym_094 python==3.6.13
conda activate gym_094
pip install gym==0.9.4
```

Please carefully read Part_B.pdf and complete the 5 questions Q1-5. 

For Q3 and Q4, please write your answer below:

Q3:Policy Iteration分为策略评估和改进两步，更关注策略的逐步改进，中间始终有策略；Value Iteration直接更新价值函数，更关注价值函数的直接优化，最后提取策略。

Q4: 策略迭代跑了146轮，价值迭代跑了15轮，价值迭代的收敛速度明显快于策略迭代。因为策略迭代需要维护和更新价值函数和策略函数两个函数，有两个两个嵌套循环；价值迭代只需要维护和更新价值函数，只有一个循环，策略是在最后一次性推导出来的。


(Q1)20 + (Q2)20 + (Q3)10 + (Q4)10 + (Q5)10 = 70 pts


Finally, compress the entire folder into a zip file (e.g. 张三_2025233111.zip) and send it to wangyc2023@shanghaitech.edu.cn