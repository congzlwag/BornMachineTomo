# BornMachineTomo
Data for Quantum Tomography with Born Machine, in support of our work [arXiv:1712.03213](https://arxiv.org/abs/1712.03213)

In this repo. there are:
1. The raw outcomes of simulated measurements on W, dimer, cluster states used in our experiments
2. The real fidelity and successive fidelity (per site) in the simulated tomography process.
	* You can convert fidelity per site $F$ to matrix distance $R$ by $R = \sqrt{(1-F^{2N})/2}$, where $N$ is the number of qubits.
3. The original data for the W30 tomography in the manuscript that illustrates similarities between tomography processes on the target state and the virtual target state.