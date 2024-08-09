import numpy as np

class LunarLander:
    g = -0.001625
    action_space = np.array([0, 1, 2, 3])
    # 0: Nothing
    # 1: Fire main engine
    # 2: Fire left-side engine
    # 3: Fire right-side engine
    thrust_map = {0: (0, 0, 0), 1: (0.001, 0, 0), 2: (0, 0.005, 0.002), 3: (0, -0.005, -0.002)}
    def __init__(self, arg=None):
        self.done = False
        self.fuel = 100
        if isinstance(arg, dict):
            self.set_state(arg)
        elif isinstance(arg, LunarLander):
            self.set_state(arg.get_state())
        elif arg is None:
            self.x, self.y = np.random.uniform(-0.75, 0.75), np.random.uniform(0.9, 1)
            self.vx, self.vy = np.random.uniform(-0.001, 0.001), np.random.uniform(-0.001, 0.001)
            self.angle, self.angular_velocity = np.random.uniform(-0.001, 0.001), np.random.uniform(-0.0005, 0.0005)
            self.surface_height = np.random.uniform(0.1, 0.2)
            self.base_x = np.random.uniform(-0.5, 0.5)
        else:
            raise TypeError("Invalid argument type. Must be LunarLander, dict, or None.")
        
    def reward(self):
        if self.y <= self.surface_height:
            self.done = True
            distance_penalty = abs(self.x - self.base_x)
            angle_penalty = abs(self.angle)
            if distance_penalty < 10 and angle_penalty < 0.1:  # Adjusted landing criteria
                reward = 10  # Successful landing
            else:
                reward = -10  # Crash
            # Adjust penalties to scale with the increased range of velocities and angles
            reward -= 1 * np.sqrt(self.vx**2 + self.vy**2)  # Velocity penalty
            reward -= 3 * angle_penalty / np.pi  # Angle penalty, increased weight
            reward -= 0.5 * distance_penalty  # Distance from base penalty, softened
        else:
            # Rewards and penalties while in air
            reward = 2 * (1 - min(abs(self.x - self.base_x), 1))  # Soften penalty, scale with init range
            reward -= 1 * (np.sqrt(self.vx**2 + self.vy**2))  # Less penalty for higher velocities
            reward -= 1 * abs(self.angle) / np.pi  # Consistent with landed angle penalty
            reward += 0.001 * self.fuel  # Assuming fuel is tracked and penalized elsewhere
        return reward

    def action_to_thrust(self, action):
        if self.fuel <= 0:
            return 0, 0, 0
        thrust, side_thrust, angular_impact = self.thrust_map.get(action, (0, 0, 0))
        if self.fuel < 10:
            thrust *= 0.5
            side_thrust *= 0.5
        return thrust, side_thrust, angular_impact

    def update(self, thrust, side_thrust, angular_impact):
        self.vx += side_thrust
        self.vy += self.g + thrust
        self.angle += self.angular_velocity
        self.angular_velocity += angular_impact
        self.x += self.vx
        self.y += self.vy
        self.fuel -= 1 if thrust > 0 or side_thrust != 0 or angular_impact != 0 else 0

    def step(self, action):
        thrust, side_thrust, angular_impact = self.action_to_thrust(action)
        self.update(thrust, side_thrust, angular_impact)
        reward = self.reward()
        return self.get_state(), reward, self.done

    def get_state(self):
        return {"position": (self.x, self.y), "velocity": (self.vx, self.vy), "angle": (self.angle, self.angular_velocity), "fuel": self.fuel, "surface_height": self.surface_height, "base_x": self.base_x}

    def set_state(self, state):
        self.x, self.y = state["position"]
        self.vx, self.vy = state["velocity"]
        self.angle, self.angular_velocity = state["angle"]
        self.fuel = state["fuel"]
        self.surface_height = state["surface_height"]
        self.base_x = state["base_x"]

    def reset(self):
        self.__init__()
        return self.get_state()

    def sample(self):
        return np.random.choice(self.action_space)