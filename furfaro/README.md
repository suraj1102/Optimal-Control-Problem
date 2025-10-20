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
- [ ] Derive the TFC formulation for the problems discussed in paper.
- [ ] Solve the double integrator problem using X-TFC and Deep-TFC.