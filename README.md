# Toward Efficient Kernel-Based Solvers for Nonlinear PDEs

Official repository for **Toward Efficient Kernel-Based Solvers for Nonlinear PDEs (ICML 2025)**.

## Environment
Our implementation is based on jax. The version we used is `jax==0.4.30`, `jaxlib==0.4.30`, `tensorly` and `optax`.
We will add a setup.py soon.

## Code Structure
The four main files are `allen_cahn/allen_cahn.py`, `burgers_pde/burgers.py`,
`eikonal_pde/eikonal.py` and `elliptic_pde/elliptic.py`. Besides Eikonal, we include configs for in each file in order
to reproduce results in paper. For Eikonal, simply just replace the num option inside run function (`run(num=18)`).

## Citation
If you find our work or code useful in your research, you could cite those with following Bibtex:
```
@misc{xu2024efficientkernelbasedsolversnonlinear,
      title={Toward Efficient Kernel-Based Solvers for Nonlinear PDEs}, 
      author={Zhitong Xu and Da Long and Yiming Xu and Guang Yang and Shandian Zhe and Houman Owhadi},
      year={2024},
      eprint={2410.11165},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.11165}, 
}
```