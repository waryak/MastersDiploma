import numpy as np


class RungeKutta:
    """
    Implementation of 4th order Runge-Kutta integration
    """

    def __init__(self, beta, rho, sigma, dt):
        self.beta = beta
        self.rho = rho
        self.sigma = sigma
        self.dt = dt

    def deriviation_step(self, initial_state, derivative, dt):
        """
        Compute one evaluation step
        """

        # evaluation of state
        state = {}

        if not derivative:
            state["x"] = initial_state["x"]
            state["y"] = initial_state["y"]
            state["z"] = initial_state["z"]
        else:
            state["x"] = initial_state["x"] + derivative["dx"] * dt
            state["y"] = initial_state["y"] + derivative["dy"] * dt
            state["z"] = initial_state["z"] + derivative["dz"] * dt

        # evaluation of derivative
        derivative_next_step = {}

        derivative_next_step["dx"] = self.sigma * (state["y"] - state["x"])
        derivative_next_step["dy"] = self.rho * state["x"] - state["y"] - state["x"] * state["z"]
        derivative_next_step["dz"] = state["x"] * state["y"] - self.beta * state["z"]

        return derivative_next_step

    def _integration_step(self, state, dt):
        """
        Runge-Kutta integration of the 4th order at a time `t` with a state `state`
        with the step `dt`
        """
        # Prepare 1,2,3,4 - order derivatives for the final "best" derivative,
        # gained as 4 first elements of the Taylor's approximation

        # Random initialization of defivatives(probably will have to moove)
        derivative = dict({"dx": np.random.normal(),
                           "dy": np.random.normal(),
                           "dz": np.random.normal()})
        rk1 = self.deriviation_step(initial_state=state,
                                    derivative=None,
                                    dt=dt * 0)

        rk2 = self.deriviation_step(initial_state=state,
                                    derivative=rk1,
                                    dt=dt * 0.5
                                    )

        rk3 = self.deriviation_step(initial_state=state,
                                    derivative=rk2,
                                    dt=dt * 0.5)

        rk4 = self.deriviation_step(initial_state=state,
                                    derivative=rk3,
                                    dt=dt)

        # When all derivatives are ready, it's time to construct Rung-Kutta derivative
        # !!!! DOUBLE CHECK THE TAYLOR'S APPROXIMATIONS !!!!
        dxdt = (1 / 6) * (rk1["dx"] + 2 * rk2["dx"] + 2 * rk3["dx"] + rk4["dx"])
        dydt = (1 / 6) * (rk1["dy"] + 2 * rk2["dy"] + 2 * rk3["dy"] + rk4["dy"])
        dzdt = (1 / 6) * (rk1["dz"] + 2 * rk2["dz"] + 2 * rk3["dz"] + rk4["dz"])

        state["x"] = state["x"] + dxdt * dt
        state["y"] = state["y"] + dydt * dt
        state["z"] = state["z"] + dzdt * dt
        return state

    def get_series(self, n_iterations, initial_state=None):
        """
        Does a series of integration steps to get a series
        :return: numpy array of series
        """
        if not initial_state:
            initial_state = dict({"x": 0.62225717,
                                  "y": -0.08232857,
                                  "z": 30.60845379})

        ode_solutions = []
        for iteration in range(0, n_iterations):
            state_t = self._integration_step(initial_state, dt=self.dt)
            ode_solutions.append(list(state_t.values()))
        result = np.array(ode_solutions)
        return result[:, 0]
