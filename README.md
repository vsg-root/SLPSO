# SLPso - Social Learning Particle Swarm Optimization

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

SLPso is a Python library that implements the Social Learning Particle Swarm Optimization (SL-PSO) algorithm for scalable optimization problems as described in the following article:

## About the Algorithm

SLPso is a Python library that implements the Social Learning Particle Swarm Optimization (SL-PSO) algorithm, which is based on the following research paper:

**A Social Learning Particle Swarm Optimization Algorithm for Scalable Optimization**
*Authors: Ran Cheng and Yaochu Jin*
*Journal: Information Sciences, Volume 291, Pages 43-60, Year 2015*
*DOI: [10.1016/j.ins.2014.08.039](https://doi.org/10.1016/j.ins.2014.08.039)*
*URL to the Paper: [Read the full paper](https://www.sciencedirect.com/science/article/pii/S0020025514008366)*

### Abstract

Social learning plays a crucial role in behavior learning among social animals. In contrast to individual (asocial) learning, social learning offers the advantage of allowing individuals to acquire behaviors from others without incurring the costs associated with individual trial-and-error experiments. The paper introduces social learning mechanisms into Particle Swarm Optimization (PSO) to develop Social Learning PSO (SL-PSO). Unlike classical PSO variants, where particles are updated based on historical information, including the best solution found by the entire swarm (global best) and the best solution found by each particle (personal best), each particle in the proposed SL-PSO learns from any superior particles (referred to as demonstrators) in the current swarm. Additionally, to simplify parameter tuning, the proposed SL-PSO adopts a dimension-dependent parameter control method. The paper evaluates SL-PSO by first comparing it with five representative PSO variants on 40 low-dimensional test functions, including shifted and rotated test functions. Furthermore, the scalability of SL-PSO is tested by comparing it with five state-of-the-art algorithms for large-scale optimization on seven high-dimensional benchmark functions (100-D, 500-D, and 1000-D). Comparative results demonstrate that SL-PSO performs well on low-dimensional problems and holds promise for solving large-scale optimization problems as well.

If you use the SLPso library in your work, please consider citing the original research paper to acknowledge the authors' contributions.



## About SL-PSO

The Social Learning Particle Swarm Optimization is a population-based optimization algorithm inspired by the behavior of a swarm of particles. It leverages social interactions to enhance exploration of the search space and convergence to optimal solutions in scalable optimization problems.

## Installation

To get started with SLPso, you can install it via pip:

```bash
pip install slpso
```
