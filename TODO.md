 - [] Fix Current Code
    - [x] Segregate Hparams logically
    - [x] Abstract f_x and g_x methods properly and define in child classes
    - [x] Abstract control input function into problem
    - [x] Rename Q_R_Matrices to make more logical sense
    - [] Abstract test_stability so that it makes logical sense
    - [x] Properly abstract computeinput wheee

    - [] Add test_stability and simulate_trajectory to base model class

 - [] Integrate with previous code
    - [] Make XTFC take problem as input
    - [x] Make Pinn take problem as input

### Plotting and Stability Stuff
- [ ] Make u[0] = 0
   - Also maybe try to have a smooth output starting from 0 and the initial state x_0
- [ ] Make plotting graph such that at t = 0 corresponds to x_0 and u_0
- [ ] Make it so time, and step size actually mean what they mean 