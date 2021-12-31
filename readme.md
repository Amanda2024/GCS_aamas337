# GCS

## Introduction 
This GIT is the implementation of the AAMAS 2022 paper 《GCS: Graph-Based Coordination Strategy for Multi-Agent Reinforcement Learning》.
In this work, we propose to factorize the joint team policy into a graph generator and graph-based coordinated policy to enable coordinated behaviours among agents.
The graph generator adopts an encoder-decoder framework that outputs directed acyclic graphs (DAGs) to capture the underlying dynamic decision structure.
We also apply the DAGness and depth constrained optimization in the graph generator to balance efficiency and performance.
The graph-based coordinated policy exploits the generated decision structure.
The graph generator and coordinated policy are trained simultaneously to maximize the discounted return. 

## Model Architecture
![image](https://user-images.githubusercontent.com/28642602/147803930-6fa3dc36-a1fd-42db-915f-ee2e04213d0a.png)


## Environments supported:

- [Multiagent Particle-World Environments (MPEs)](https://github.com/openai/multiagent-particle-envs)
- Collaborative Gaussian Squeeze
- [Google Research Football](https://github.com/google-research/football)



##  Installation

``` Bash
# create conda environment
conda create -n marl python==3.6.1
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```
# install onpolicy package
cd GCS
pip install -e .
pip install -r requirements.txt
```

## Training
1. we use train_guassian.py as an example:

```
cd onpolicy/scripts/train
python train_guassian.py
```

2. we use train_mpe.py as an example:

```
cd onpolicy/scripts/train
python train_mpe.py
```

3. we use train_football.py as an example:

```
cd onpolicy/scripts/train
python train_football.py
```




## Citing GCS

If you use GCS in your research, or any other implementation provided here, please cite the GCS paper.



