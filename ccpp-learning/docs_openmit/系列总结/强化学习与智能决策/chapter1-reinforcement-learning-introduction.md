title: "第01章：强化学习介绍" 
mathjax: true
date: 2017-08-03 22:43:49
categories: 
	- 强化学习与智能决策
tags: 
	- RL
	- Agent
	- Environments
---

+ author: zhouyongsdzh@foxmail.com
+ date: 2016-08-10
+ weibo: [@周永_52ML](http://weibo.com/p/1005051707438033/home?)

## 符号定义

| 符号 |  物理意义 |
| --- | :-- |
| $\mathcal{S}$ | （有限的）状态集合 |
| $\mathcal{A}$ | （有限的）动作集合 |
| $\mathcal{P}$ | 状态转移概率 |
| $R$ | 奖赏 Reward |
| $\gamma$ | 折扣因子，$\gamma \in [0,1]$
| $S_t$ | $t$时刻的状态 ($S_t \in \mathcal{S}$) |
| $A_t$ | $t$时刻的动作 ($A_t \in \mathcal{A}$) |
| $\pi(a\|s)$ | 策略函数，状态到动作的映射 |
| $G_t$ | $t$时刻开始回报序列，即$G_t=R_{t+1} + \gamma R_{t+2} + ...$ |
| $v_{\pi}(s)$ | 状态值函数，在策略$\pi$下状态$s$的值函数 |
| $q_{\pi}(s,a)$ | 状态值函数，在策略$\pi$下状态$s$采取动作$a$值函数 |
