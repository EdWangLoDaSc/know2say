好，我先 **double-check一下整个 setting**，然后把 **advantage 的计算 pipeline 写成一个可以直接实现的步骤（接近算法 / code level）**。我会把关键变量和每一步计算都明确出来，这样你可以直接放进 RL loop 里。

下面假设我们是在 **RLVR / GRPO-style training** setting。

---

# 1. Training Setting（先把问题定义清楚）

## Environment

每个训练样本：

[
x \sim D
]

模型生成 reasoning trajectory：

[
\tau_i = (y_{i1},y_{i2},...,y_{iT_i})
]

完整 sequence：

[
s_{it} = (x,y_{i1:t})
]

---

## Group Sampling（GRPO style）

每个 prompt sample **G trajectories**

[
\mathcal{G}(x)={\tau_1,...,\tau_G}
]

terminal reward（verifier）：

[
r_i \in {0,1}
]

---

## Checkpoints（不是每 token）

我们只在一部分 prefix 上计算 reachability：

[
C_i = {t_1,t_2,...,t_K}
]

比如：

* branch divergence
* entropy spike
* equation emission
* code statement end
* reasoning delimiter

定义：

[
c(t) = \text{最近的 checkpoint}
]

---

# 2. Reachability Potential Estimation

对于 prefix：

[
s_{it}=(x,y_{i1:t})
]

做 **probe continuation**

sample (M) short rollouts：

[
z_{it}^{(1)},...,z_{it}^{(M)}
]

completion：

[
y_{it}^{(m)} = (y_{i1:t},z_{it}^{(m)})
]

verifier：

[
v(y_{it}^{(m)}) \in {0,1}
]

估计 reachability：

[
\hat{\Phi}(s_{it}) =
\frac{1}{M}
\sum_{m=1}^{M} v(y_{it}^{(m)})
]

直觉：

[
\hat{\Phi}(s) = P(\text{can still solve}|prefix)
]

---

# 3. Potential Transform

直接用 probability 会不稳定，所以用 **logit potential**

[
\Psi(s)=\text{logit}(\hat{\Phi}(s))
]

[
\Psi(s)=
\log
\frac{\hat{\Phi}(s)+\epsilon}
{1-\hat{\Phi}(s)+\epsilon}
]

优点：

* small probability differences 不会太弱
* matches policy gradient scale

---

# 4. Step Advantage Decomposition

我们把 advantage 拆成两部分：

trajectory-level + reachability increment

---

## (A) GRPO trajectory advantage

group mean reward：

[
\bar r =
\frac1G\sum_{j=1}^{G} r_j
]

trajectory credit：

[
A^{traj}_i = r_i - \bar r
]

这是 standard GRPO。

---

## (B) Reachability Increment Advantage

对于 checkpoint：

[
A^{reach}_{i,t_k}
=================

## \Psi(s_{i,t_k})

\Psi(s_{i,t_{k-1}})
]

解释：

* 如果 prefix 让 problem 更 **可解**
* potential increase
* advantage positive

如果 prefix 把 reasoning 带入 **dead-end**

potential drop
advantage negative

---

## Token-level broadcast

checkpoint 之间的 token 共享 credit

对于 token (t)

找到 segment：

[
t_k \le t < t_{k+1}
]

则

[
A^{reach}_{i,t}
===============

A^{reach}_{i,t_k}
]

---

# 5. Final Advantage Formula

最终 advantage：

[
A_{i,t}
=======

(1-\alpha)A^{traj}*i
+
\alpha A^{reach}*{i,t}
]

其中

[
\alpha \in [0,1]
]

控制 step credit 权重。

---

## Practical schedule

训练 early：

[
\alpha \approx 0.2
]

训练 middle：

[
\alpha \approx 0.5
]

late：

[
\alpha \approx 0.7
]

原因：

early prefix noisy
later reasoning structure稳定。

---

# 6. All-Incorrect Group Handling

这是最关键的。

如果

[
r_i=0 \quad \forall i
]

则

[
A^{traj}_i = 0
]

但 reachability 仍然存在：

例如：

| prefix | Φ    |
| ------ | ---- |
| step1  | 0.65 |
| step2  | 0.72 |
| step3  | 0.70 |
| step4  | 0.04 |

则

[
A^{reach}_2>0
]

[
A^{reach}_4<<0
]

模型会 learn：

* step1-3 good reasoning
* step4 catastrophic error

GRPO / DAPO 在这里完全没 signal。

---

# 7. PPO / GRPO Training Loss

用 importance sampling：

ratio：

[
\rho_{i,t}
==========

\frac
{\pi_\theta(y_{it}|s_{i,t-1})}
{\pi_{old}(y_{it}|s_{i,t-1})}
]

loss：

[
L
=

*

\mathbb E
\Big[
\min(
\rho A,
\text{clip}(\rho,1-\epsilon,1+\epsilon)A
)
\Big]
]

其中：

[
A=A_{i,t}
]

---

# 8. Compute Optimization（非常关键）

不要每 prefix rollout。

只对以下 prefix probe：

### 1 entropy spikes

[
H(\pi(\cdot|s_t)) > \tau
]

---

### 2 prefix divergence

prefix tree branch point

---

### 3 reasoning boundaries

比如：

```
Step 3:
```

---

### 4 near terminal reasoning

final equation / boxed answer

---

这样：

probe cost

[
O(G K M)
]

而不是

[
O(G T M)
]

通常

[
K << T
]

---

# 9. Algorithm Summary

## RAPO Advantage Computation

```
for prompt x:

    sample G trajectories τ_i

    compute rewards r_i

    detect checkpoints C_i

    for each checkpoint t_k:

        run M probe continuations

        estimate Φ(s_{t_k})

        compute potential Ψ

    for each checkpoint:

        A_reach = Ψ(s_k) - Ψ(s_{k-1})

    broadcast A_reach to tokens

    compute A_traj = r_i - mean(r)

    final advantage:

        A = (1-α)A_traj + α A_reach
```

---

# 10. Why this is theoretically clean

关键性质：

### Potential shaping property

如果

[
\Phi(s)
=======

P(\text{success}|s)
]

exact。

则

[
\sum_t A^{reach}_{i,t}
======================

\Psi(s_T)-\Psi(s_0)
]

因为

[
\Psi(s_T)=\logit(r_i)
]

整个 reward decomposition 是 **consistent with terminal objective**。

不会引入 policy bias。

---

# 11. Reviewer会问的三个问题（提前准备）

### Q1 probe policy bias

probe 不同于 training policy。

答：

* probe 是 **reachability estimator**
* 类似 value function oracle
* 不参与 gradient

---

### Q2 probe cost

答：

checkpoint gating

uncertainty-based probing

只在 ambiguous states compute。

---

### Q3 verifier noise

答：

use **bootstrapped Φ smoothing**

[
\tilde{\Phi} = \lambda Φ_{probe} + (1-\lambda)Φ_{group}
]

减少 variance。

---

# 12. 一句话总结这个 advantage

核心公式：

[
A_{i,t}
=======

(1-\alpha)(r_i-\bar r)
+
\alpha
\big(
\Psi(s_{c(t)})-\Psi(s_{p(c(t))})
\big)
]

含义：

**reward not for being right — reward for preserving solvability.**

---

如果你愿意，我可以再帮你把这个 **升级成一个真正可以打 top-tier 的版本**，因为现在这个其实还有两个可以明显增强的地方：

1️⃣ **把 Φ learning 成一个 reusable reachability model**（不用每次 probe）
2️⃣ **把 credit assignment 写成一个 theorem（policy invariant shaping）**

这两个一加，paper level 会明显从 *nice idea* → *serious contribution*。


这个问题其实很关键，因为 **checkpoint 选错了，整个方法会变成 noisy heuristic**。我建议从“**理论合理性 + 实现稳定性 + reviewer 接受度**”三个维度看。结论先说：

**最合理的其实是两类 checkpoint 的组合：**

> **(1) prefix divergence points + (2) entropy spikes**

其它几个（equation emission / delimiter / code statement）更适合作为 **task-specific optional triggers**。

下面我仔细分析一下。

---

# 1️⃣ Prefix Divergence（最核心、最 principled）

**定义**

如果 group 中两条 trajectory 在 t 处产生不同 token：

[
y_{i,t} \neq y_{j,t}
]

则

[
t \in C
]

也就是 **prefix tree 的分叉点**。

---

## 为什么这个最合理

因为 credit assignment 的本质是：

> **什么时候不同选择导致不同结果**

分叉点正好就是这个位置。

在 prefix tree 上：

```
prefix
  |
 step t
  |
  +--- token A → 成功
  |
  +--- token B → 失败
```

这里：

[
\Phi(s_t^A) \gg \Phi(s_t^B)
]

所以

[
A^{reach}_t
]

非常 meaningful。

---

## reviewer也很喜欢这个

因为：

* TEMPO already uses prefix tree
* VinePPO also reasons on prefix state
* 很 natural extension

不会被说 heuristic。

---

# 2️⃣ Entropy Spike（第二个最合理）

定义：

[
H(\pi(\cdot|s_t)) > \tau
]

例如：

top-k entropy threshold。

---

## 为什么 entropy spike 合理

因为它表示：

> **模型在做 decision**

很多 reasoning 关键步骤是这种：

```
Step 4:
Now we need to compute...

(a+b)^2
```

模型在这里 uncertainty 高。

如果这一步走错：

后面可能 irrecoverable。

---

## entropy 的另一个优点

它解决 **group 不够多的问题**。

如果

[
G=4
]

prefix divergence 很少。

entropy 可以补充 detect important decision。

---

# 3️⃣ Equation Emission（数学任务 useful，但不 universal）

例如：

```
x = 3
```

或者

```
Therefore the answer is 42
```

优点：

* reasoning boundary clear
* step-level semantics

问题：

* task specific
* NLP tasks 没用

所以：

我建议把它作为

**dataset-specific checkpoint plugin**

---

# 4️⃣ Code Statement End（代码任务 useful）

例如：

```python
for i in range(n):
```

或者

```python
return x
```

因为 program state 在 statement 级别变化。

但同样问题：

* domain specific

---

# 5️⃣ Reasoning Delimiter（最弱）

例如：

```
Step 1:
Step 2:
```

问题：

* delimiter 是 prompt engineering
* 很多模型不稳定

reviewer 很可能会说：

> "this relies on formatting artifacts"

所以不推荐作为 primary signal。

---

# 推荐最终 checkpoint design

我建议用 **two-level gating**。

---

# Level 1: Universal checkpoints

必须触发

### ① Prefix divergence

[
y_{i,t} \neq y_{j,t}
]

---

### ② Entropy spike

[
H(\pi(\cdot|s_t)) > \tau
]

---

# Level 2: Task-specific checkpoints（optional）

如果检测到：

### math

* equation emission
* boxed answer

### code

* AST boundary
* statement end

---

# 最终 checkpoint 集合

[
C =
C_{divergence}
\cup
C_{entropy}
\cup
C_{task}
]

---

# checkpoint 数量控制

否则 compute 会爆。

只保留

[
|C| \le K_{max}
]

例如：

```
K_max = 6
```

选择：

score top-K。

score 定义：

[
score(t) =
\lambda_1 \text{branch_variance}
+
\lambda_2 H(\pi)
]

---

# 更干净的数学定义（论文可以写）

定义 decision score：

[
D(s_t)
======

\alpha Var_{i\in G}[y_{i,t}]
+
\beta H(\pi(\cdot|s_t))
]

checkpoint：

[
C =
\text{TopK}_t D(s_t)
]

---

# 为什么这个 reviewer 很难 attack

因为：

* divergence = **empirical decision**
* entropy = **policy uncertainty**

两者结合就是：

> states where the model both **disagrees across rollouts and is uncertain about its action**

这正是 credit assignment 应该关注的位置。

---

# 一个 reviewer 很喜欢的说法

论文可以写：

> We place reachability probes at *decision states*—prefixes where the policy either exhibits high uncertainty or produces divergent continuations across sampled trajectories.

---

# 最后给你一个我的真实建议

如果是 **top-tier idea**，我会这样设计：

checkpoint 只用：

[
C = C_{divergence} \cup C_{entropy}
]

理由：

* simplest
* most principled
* cross-domain
* reviewer 最难挑毛病

其它全部作为 **appendix ablation**。


