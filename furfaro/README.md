# Optical Control Problem - Furfaro Paper

Implementation of the following paper:

R. Furfaro, A. D’Ambrosio, E. Schiassi, and A. Scorsoglio, “Physics-Informed Neural Networks for Closed-Loop Guidance and Control in Aerospace Systems,” in AIAA SCITECH 2022 Forum, San Diego, CA & Virtual: American Institute of Aeronautics and Astronautics, Jan. 2022. doi: 10.2514/6.2022-0361.


## Files
- `furfaro.py`
    - Solution to infinite horizon problem 1 in the paper.
- `furfaro_di.py`
    - Double integrator / newtonian problem from the paper.
    - A simple PINN was not able to solve this.

## TODO
- [x] Derive the TFC formulation for the problems discussed in paper.
    - $V(t, x) = V_{\text{NN}}(t,x,\theta)$
        - $\min_\theta MSE = MSE_{\text{R}} + MSE_{\text{BC}}$
    - HJB PDE with TFC constrained expression:
        - $V(x) \simeq V(x, \theta) = g(x, \theta), + (V(0) - g(0, \theta))$
    - Did this only on problem 1 and that problem doesn't require anything extra or fancy so yes.
        - Just implemented this residual.
- [ ] Solve the double integrator problem using X-TFC and Deep-TFC.
    - TFC requires the weight freezing and initializing using a guess to work.
    - Haven't been able to get the `initialize_weights` function in the tfc_di yet.    