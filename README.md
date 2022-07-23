## Simplistic Federated Learning Simulator
This repo was created to provide a good base for experimenting
with federated learning methods with minimal requirements and complexity. 

This repo contains a PyTorch implementation of Generalized FedAvg 
([Algorithm 1](https://arxiv.org/pdf/2107.06917.pdf)), where the user
has several degrees of freedom in terms of algorithmic design. 
In particular, the user can define:

- global and local optimizers (SGD is implemented)
- aggregation scheme (simple mean is implemented)
- client sampling (uniform sampling is implemented)

The source code is provided in the [fl_sim](fl_sim) directory. The user can extend the 
usability by implementing new aggregation schemes ([fl_sim/aggregation](fl_sim/aggregation)),
client samplings ([fl_sim/client_sampling](fl_sim/client_sampling)), or by bringing in new 
models ([fl_sim/models](fl_sim/models) and datasets ([fl_sim/data_funcs](fl_sim/data_funcs). 
The user can also define a new task in [tasks.py](fl_sim/utils/tasks.py). The repo also 
contains two notebooks. [The first notebook](notebooks/generate_experiments.ipynb)
contains an example of a simple task, where it 
generates the script to run experiments via terminal. [The second notebook](notebooks/plot_results.ipynb))
provide means to visualize obtained results. 

### More Advanced FL Frameworks

For more advanced FL simulators based on PyTorch, we recommend 

- FL_Pytorch: https://github.com/burlachenkok/flpytorch
- FLSim: https://github.com/facebookresearch/FLSim
- fl-simulation: https://github.com/microsoft/fl-simulation

If you are looking for a framework that goes beyond simple simulations, we recommend 

- Flower: https://flower.dev/
- FedML: https://github.com/FedML-AI