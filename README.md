
# PPO with custom lander environment
![Screenshot 2025-05-31 032530](https://github.com/user-attachments/assets/4eef797e-f016-47cf-af04-0f8900129b01)

Recently, I have been very interested in reinforcement learning, so I wanted to create a project around that. I built a custom lander game using pygame-ce and used the PPO implementation from stable-baseline-3. This is still a work in progress, but the manual mode and training are currently working.




## Installation

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/ZeroMeOut/PPO-with-custom-lander-environment.git](https://github.com/ZeroMeOut/PPO-with-custom-lander-environment.git)
    cd PPO-with-custom-lander-environment
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


    
## Usage
    ```bash
    python main_menu.py
    ```
The manual mode for you to play the game, and the training to train the lander. The testing is for well, testing, but still a work in progress




## Acknowledgements

 - [Sent's video](https://www.youtube.com/watch?v=uKnjGn8fF70&t=540s)
 - [Baraltech's main menu for pygame](https://github.com/baraltech/Menu-System-PyGamee)
 - [This very wonderful blog](https://antoinebrl.github.io/blog/rl-mars-lander/#reward-shaping)
 - [Phildecroos's lunar lander game](https://github.com/phildecroos/lunar_lander/)
 - [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/s/GGCzYxGnLp)
 


## Random Thoughts
First off, I was initially trying to use CNN to train this. But it's so damn stressful, I think the default encoder used by stable baseline might not be the greatest for this. I would try to make a custom one someday.

Secondly, it is annoyingly long to fine-tune and train. I took like 20-25 tries to get a decently looking ep_rew_mean graph, and even then I kept running into [value loss explosion](https://medium.com/@kaige.yang0110/in-training-ppo-how-to-balance-value-loss-and-policy-loss-cbf10d9d6b86) because of the hyperparameters or rewards I was using (thank you Antoinebrl, you saved my time a lot). And lord, the bugs, I don't even want to start with that.

Overall, this is still a learning process for me, and even with all the rambling I am doing rn I still find building this fun.
