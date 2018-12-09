# BornMachineTomo
Data for Quantum Tomography with Born Machine, in support of our work [arXiv:1712.03213](https://arxiv.org/abs/1712.03213)

## File Organization
- Definitions of the classes `./CS6.py`
- Efficiency experiments on typical states `./trial8-Efficiency/`
  - `main.py` conducts the experiments
  	- Using measurement outcomes in `./MeasOutcomes/`
  - Resulting fidelity sequences are in the folders `[typ]/[N]/`
  - `sat_persite.py` postprocess the fidelity sequeces to analyze what if we set the per-site fidelity as criterion.
- Efficiency experiments on random states `./trial14-RandTarget/
  `
  - `main.py` conducts the experiments
  	- Using measurement outcomes in `./MeasOutcomes/Random/`
  - Resulting fidelity sequences are in the folders `[Dmax]/[N]/`
- Demonstration of Fidelity Estimation `./trial13-FidEstL249`
  - `prep.py` prepares measurement outcomes from the virtual target states stored in `./trial8-Efficiency/[type]/[N]/R[seed]/L249/` and `./trial9-randomTarget/random/[N]/[seed]/R[seed]/L249/` and stores the outcomes in `vir_measout/`
  - `main.py` conducts the experiments
    - Using measurement outcomes in `vir_measout/`
  - Resulting fidelity sequences are in the folders `[type]/[N]/`
- Robustness Experiments `./errRobust/`
  - `prep.py` prepares measurement outcomes from the noised target states $\sigma_\epsilon = (1-\epsilon)\sigma + \frac{\epsilon}{q^N}\mathrm{I}$ and stores them in `./errRobust/MeasOutcomes/`
  - `main.py` conducts the experiments
    - Using measurement outcomes in `./errRobust/MeasOutcomes/[type]/[N]/[noise]/`
  - Resulting fidelity sequences are in the folders `[type][N]/[noise]/`
  - `elist.npy` includes the values of noisy level we considered.
- `MeasOutcomes/` includes raw outcomes of simulated measurements. 
  - `prep.py` measures the typical states and stores the results in:
  -  `[type]/[N]/R[seed]Set.pickle`, which pickles the outcomes from the state of `[type]` and length  `[N]` in the random case initiated by `[seed]`, the state being stored as  `[type]/[N]/stdmps.pickle`
  - `Random/`
    - `prep_Rand.py` measures the random states and stores the results in:
    -  `[Dmax]/[N]/[seed]/R[seed]Set.pickle`, which pickles the outcomes from the  random state (by `[seed]`) whose Dmax is `[Dmax]` and length is `[N]` in the random case initiated by `[seed]`, the state being stored as  `[type]/[N]/[seed]/stdmps.pickle`
- `WorkSpace.ipynb` is the Jupyter Notebook where the results are analyzed and plotted

## Note

* Due to the big volume of the outcomes, we only uploaded some of the training sets we used in the numerical experiments mentioned in our manuscript, yet the scripts `pre*.py` suffice the generation of all the training data.