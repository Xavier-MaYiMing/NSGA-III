### NSGA-III: Nondominated sorting genetic algorithm III

##### Reference: K. Deb and H. Jain, An evolutionary many-objective optimization algorithm using reference-point based non-dominated sorting approach, part I: Solving problems with box constraints, IEEE Transactions on Evolutionary Computation, 2014, 18(4): 577-601.

##### NSGA-III is an improved version of the classic multi-objective evolutionary algorithm (MOEA) NSGA-II. NSGA-III implements reference points to tackle the difficulties of many-objective optimization.

| Variables | Meaning                                              |
| --------- | ---------------------------------------------------- |
| npop      | Population size                                      |
| iter      | Iteration number                                     |
| lb        | Lower bound                                          |
| ub        | Upper bound                                          |
| nobj      | The dimension of objective space                     |
| pc        | Crossover probability (default = 1)                  |
| pm        | Mutation probability (default = 1)                   |
| eta_c     | Spread factor distribution index (default = 30)      |
| eta_m     | Perturbance factor distribution index (default = 20) |
| nvar      | The dimension of decision space                      |
| pop       | Population                                           |
| objs      | Objectives                                           |
| V         | Reference vectors                                    |
| zmin      | Ideal points                                         |
| rank      | Pareto rank                                          |
| pf        | Pareto front                                         |

#### Test problem: DTLZ1

$$
\begin{aligned}
	& k = nvar - nobj + 1, \text{ the last $k$ variables is represented as $x_M$} \\
	& g(x_M) = 100 \left[|x_M| + \sum_{x_i \in x_M}(x_i - 0.5)^2 - \cos(20\pi(x_i - 0.5)) \right] \\
	& \min \\
	& f_1(x) = \frac{1}{2}x_1x_2 \cdots x_{M - 1}(1 + g(x_M)) \\
	& f_2(x) = \frac{1}{2}x_1x_2 \cdots (1 - x_{M - 1})(1 + g(x_M)) \\
	& \vdots \\
	& f_{M - 1}(x) = \frac{1}{2}x_1(1 - x_2)(1 + g(x_M)) \\
	& f_M(x) = \frac{1}{2}(1 - x_1)(1 + g(x_M)) \\
	& \text{subject to} \\
	& x_i \in [0, 1], \quad \forall i = 1, \cdots, n
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(91, 400, np.array([0] * 7), np.array([1] * 7))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/NSGA-III/blob/main/Pareto%20front.png)



