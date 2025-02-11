from tensorflow.keras import models, layers, optimizers
from collections import deque
import numpy as np
import random 

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
    #Esta función es la red neuronal, que resulta el aproximador de la función Q que mostramos arriba
    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='softmax'))  # Softmax para probabilidades
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001))
        return model
    #En esta función de memoria guardamos el estado s, el siguiente s', r y a.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Exploración aleatoria
        probs = self.model.predict(state)[0]  # Probabilidades predichas. Usamos maxQ(s,a)
        return np.random.choice(self.action_size, p=probs)  # Muestreo probabilístico

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_probs = self.model.predict(next_state)[0] #Representa Q(s',a')
                target = reward + self.gamma * np.max(next_probs) #Representa r + γ * maxQ(s',a')
            target_f = self.model.predict(state) #Representa Q(s,a)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay