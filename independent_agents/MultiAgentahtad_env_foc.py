import gym
import os
import json
import matplotlib
import numpy as np
np.set_printoptions(linewidth=np.inf)
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('tkAgg')
plt.style.use("seaborn")
import matplotlib.colors as mcolors
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from gym import spaces
from collections import deque
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class MultiAgentahtadEnv_FOC(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        print('active hypothesis testing for anomaly detection activated')
        self.report_schedule = 1000
        self.num_agents=kwargs.get('num_agents', None)
        self.good_episodes = {agn:0 for agn in range(self.num_agents)}
        self.inactive_agents_array=None
        self.add_actions_to_obs=kwargs.get('add_actions_to_obs', False)
        self.coupled_reward=kwargs.get('coupled_reward', None) # enable coupled reward
        self.save_path_inference=kwargs.get('save_path_inference', None)
        self.info_interval=20000
        self.mode = kwargs.get('mode', None)

        self.n_processes = kwargs.get('n_processes', None)
        if self.add_actions_to_obs:
            print('adding actions to observations')
        self.one_hot = np.eye(self.n_processes+1)
        self.one_hot[-1] = np.full(self.n_processes+1,0)
        self.one_hot = self.one_hot[:,:-1]

        self.win_reward = kwargs.get('win_reward', None)
        self.lose_reward = kwargs.get('lose_reward', None)
        self.step_cost = kwargs.get('step_cost', None)
        self.step_cost_ = -1.0
        self.punishment = kwargs.get('punishment', None)
        self.action_space = spaces.Discrete(self.n_processes+1)
        self.observation_space = spaces.Box(low=0.0,high=1.0,shape=(self.n_processes+(self.num_agents-1)*(self.n_processes)*self.add_actions_to_obs,),dtype=np.float32)
        self.sigma = kwargs.get('sigma', None)
        self.eps = kwargs.get('eps', 1E-27)
        self.individual_hypotheses = pd.DataFrame([], columns=['hypothesis_' + str(i + 1) for i in range(self.n_processes)])
        self.processes = {} #pd.DataFrame([], columns=['process_' + str(i + 1) for i in range(self.n_processes)])
        self.episode_number = 0
        self.time = {}
        self.time_max = list([0,0])
        self.state=None

        self.p_abnormal = np.full(self.n_processes, 1 / self.n_processes)

        self.n_hypotheses=self.n_processes
        self.available_hypothesis=None
        self.generate_source_hypotheses()
        self.belief = None
        self.process_belief = None
        self.reward={}
        self.dones = set()
        self.actions_count = 0
        self.actions_taken = []
        self.previous_actions_count = 0
        self.pref=dict(success=[], steps=[])
        self.hypes_lengths = {}
        self.hypes_successes = {}
        self.no_conv = {agn:dict((str(k), 0) for k in self.available_hypothesis.index.values) for agn in range(self.num_agents)}

        self.success_array = {}
        self.reward_array = {}
        self.internal_reward_array = {}
        self.length_array = {}
        self.hypes_lengths = {}
        self.hypes_successes = {}
        self.agents_correct={}

        self.no_convergence_counter={agn:0 for agn in range(self.num_agents)}
        self.no_convergence_counter_train=0
        self.no_convergence_success={agn:[] for agn in range(self.num_agents)}

        self.train_info=[]
        self.global_time=0
        if self.mode != 'inference': plt.switch_backend('Agg')

        for i in range(self.num_agents):
            self.reward_array[i]= deque()
            self.internal_reward_array[i]= deque()
            self.success_array[i] = deque()
            self.length_array[i] = deque()
            self.hypes_lengths[i] = dict((str(k), []) for k in self.available_hypothesis.index.values)
            self.hypes_successes[i] = dict((str(k), []) for k in self.available_hypothesis.index.values)

        self.done_report = {agn:True for agn in range(self.num_agents)}
        self.horizon_punishment = kwargs.get('horizon_punishment', None)

    def generate_source_hypotheses(self):
        source_hypothesis = np.zeros(shape=self.n_hypotheses, dtype=np.int)
        source_hypothesis[0] = 1
        source_hypothesis = source_hypothesis[np.newaxis, :]
        for _ in range(self.n_hypotheses - 1):
            source_hypothesis = np.vstack([source_hypothesis, np.roll(source_hypothesis[-1], 1)])
        self.available_hypothesis = pd.DataFrame(source_hypothesis, columns=['process_' + str(i + 1) for i in range(self.n_processes)])

    def reset(self):
        self.done_report = {agn:True for agn in range(self.num_agents)}
        self.internal_reward_array = {}
        self.belief = {}
        self.process_belief = {}
        self.actions_count = {}
        self.actions_taken = {}
        self.reward = {}
        self.processes={}
        self.agents_correct={}
        self.init_hypothesis()

        for i in range(self.num_agents):
            self.internal_reward_array[i] = deque()
            self.belief[i] = [np.repeat(1 / self.n_hypotheses, self.n_hypotheses)]
            self.process_belief[i] = [np.repeat(1 / self.n_processes, self.n_processes)]
            assert np.isclose(np.sum(self.belief[i]), 1.0, atol=1e-4), f'init belief - {self.belief[i]}, {self.process_belief[i]}'
            assert np.isclose(np.sum(self.process_belief[i]), 1.0, atol=1e-4), f'init belief - {self.belief[i]}, {self.process_belief[i]}'

            self.actions_count[i]=0
            self.actions_taken[i]=[]
            self.time[i] = 0
            self.reward[i] = []
            self.processes[i]=pd.DataFrame([], columns=['process_' + str(i + 1) for i in range(self.n_processes)])
        self.episode_number+=1
        self.dones = set()
        self.dones_array = np.repeat(False,self.num_agents)
        self.agents_correct = np.repeat(False, self.num_agents)

        if self.add_actions_to_obs:
            return {key:np.concatenate([np.squeeze(value),np.tile(self.one_hot[-1],self.num_agents-1)]) for key,value in self.belief.copy().items()}
        else:
            return {key:np.squeeze(value) for key,value in self.belief.copy().items()}

    def init_hypothesis(self):
        h_index = np.random.choice([ii for ii in range(self.available_hypothesis.shape[0])], p=self.p_abnormal)
        self.individual_hypotheses = self.available_hypothesis.iloc[h_index].copy()

    def step(self, action_dict):
        self.global_time+=1
        obs, rew, done, info = {o:np.zeros(shape=self.n_hypotheses+(self.num_agents-1)*(self.n_processes)*self.add_actions_to_obs,) for o in range(self.num_agents)}, \
                               {o:0.0 for o in range(self.num_agents)}, {}, {}

        for i, action in action_dict.items():
            if self.dones_array[i]: continue
            self.get_new_sample(agent=i)
            self.actions_taken[i].append(action)

            if self.time[i] > self.horizon_punishment :
                action=self.n_processes
                if self.mode=='inference':
                    self.no_convergence_counter[i] += 1
                    self.no_conv[i][str(self.individual_hypotheses.name)]+=1
                    self.no_convergence_success[i].append(np.argmax(self.belief[i][-1]) == self.individual_hypotheses.name)
                if self.mode!='inference': self.no_convergence_counter_train += 1

            if action==self.n_processes:
                self.time[i] += 1
                if not self.time[i] > self.horizon_punishment:
                    self.success_array[i].appendleft(np.argmax(self.belief[i][-1]) == self.individual_hypotheses.name)
                    self.length_array[i].appendleft(self.time[i])
                self.belief[i].append(self.belief[i][-1])
                self.dones.add(True)
                self.dones_array[i]=True
                self.time_max= list([self.time[i],self.episode_number]) if self.time[i]>self.time_max[0] else self.time_max

                if np.argmax(self.belief[i][-1]) == self.individual_hypotheses.name:
                    self.agents_correct[i]=True
                    self.reward[i].append(0.0 if self.time[i]<self.horizon_punishment else self.punishment)
                else:
                    self.reward[i].append(0.0 if self.time[i]<self.horizon_punishment else self.punishment)

                if self.mode!='inference': self.train_info.append([f'{self.time[i]}', f'{np.max(self.belief[i])}', f'{np.argmax(self.belief[i][-1])}',f'{self.individual_hypotheses.name}'])

                if self.episode_number%100000==0 or self.episode_number in [100,1000]:
                    try:
                        self.render(mode=self.mode,save_path=self.save_path_inference,episode_number=self.episode_number, agent=i)
                    except:
                        print('render error')
            else:
                self.update_exclusive_belief_by_action(action=action,i_belief=i)

                self.reward[i].append(self.step_cost)
                self.time[i] += 1

            actions_holder=np.full(self.num_agents,self.n_processes)
            if len(action_dict)==self.num_agents:
                for jj,(k,x) in enumerate(action_dict.items()):
                    actions_holder[k]=x
                actions_holder = np.delete(actions_holder, i)
            else:
                actions_holder=self.one_hot[-1]
                assert False, 'number of actions is less than number of agents'

            shared_obs=np.concatenate([self.belief[i][-1].copy(),*self.one_hot[actions_holder]])

            obs[i], rew[i], done[i], info[i] = shared_obs if self.add_actions_to_obs else self.belief[i][-1].copy(), self.reward[i][-1], False, self.actions_taken.copy()

        if self.add_actions_to_obs:
            actions_holder = np.full(self.num_agents, self.n_processes)
            if len(action_dict) == self.num_agents:
                for jj, (k, x) in enumerate(action_dict.items()):
                    actions_holder[k] = x

            for don_idx, don in enumerate(self.dones_array):
                if don:
                    if self.done_report[don_idx]:
                        self.done_report[don_idx] = False
                    else:
                        if abs(self.time[abs(1-don_idx)] - self.time[don_idx]) < 100:
                            obs[abs(1-don_idx)][-self.action_space.n+1:] = self.one_hot[np.argmax(self.belief[don_idx][-1])]
                        else:
                            obs[abs(1-don_idx)][-self.action_space.n+1:] = np.full_like(self.one_hot[0],fill_value=1)


        done["__all__"] = np.sum(self.dones_array) == self.num_agents

        if done["__all__"] and self.mode=='inference':
            if self.episode_number%1000==0: print('**********************TEN REWARD**********************')
            ten_reward = self.win_reward*np.sum(self.agents_correct)/self.num_agents if any(self.agents_correct) else self.lose_reward
            for ri in range(self.num_agents):
                self.reward[ri][-1]=ten_reward
        if self.coupled_reward:
            if done["__all__"]:
                # reward_sum=np.sum(self.agents_correct)*self.win_reward if any(self.agents_correct) else self.lose_reward
                if all(self.agents_correct):
                    reward_sum = 1200
                elif any(self.agents_correct):
                    reward_sum = 470
                elif not any(self.agents_correct):
                    reward_sum = -60
                if any(np.array(list(self.time.values()))>=self.horizon_punishment):
                    reward_sum=self.punishment
                for agn in range(self.num_agents):
                    if not self.time[i] > self.horizon_punishment:
                        self.reward_array[agn].appendleft(reward_sum+np.sum(self.internal_reward_array[agn]))
                for i in range(self.num_agents):
                    if self.episode_number % self.report_schedule == 0 and self.episode_number != 0:
                        print(
                        f'before - ({len(self.success_array[i])},{len(self.length_array[i])},{len(self.reward_array[i])}), ',
                        end='')
                        try:
                            success_array_limited = [self.success_array[i].pop() for _ in range(len(self.success_array[i]))]
                            length_array_limited = [self.length_array[i].pop() for _ in range(len(self.length_array[i]))]
                            reward_mean = [self.reward_array[i].pop() for _ in range(len(self.reward_array[i]))]

                            print(f'agent-{i} | success rate - {np.sum(success_array_limited) / len(success_array_limited)}, '
                                  f'episode average length - {np.mean(length_array_limited,dtype="float64")}, average reward - {np.mean(reward_mean)} '
                                  f'after - ({len(self.success_array[i])},{len(self.success_array[i])},{len(self.reward_array[i])}) | episode - {self.episode_number} | timesteps - '
                                  f'{self.global_time} | step cost - {self.step_cost} | N.E - {self.no_convergence_counter_train}')
                        except:
                            print('pop from array didn''t succeeded')
            else:
                reward_sum=np.sum(list(rew.values()))
                for agn in range(self.num_agents): self.internal_reward_array[agn].append(reward_sum)
            rew = dict.fromkeys(rew,reward_sum)

        return obs, rew, done, info

    def get_new_sample(self,agent):
        self.processes[agent].loc[self.time[agent], :] = np.random.normal(self.individual_hypotheses, self.sigma)

    def update_exclusive_belief_by_action(self, action, i_belief):
        assert action>=0,'action is lower than 0'
        assert action<=self.n_processes-1,f'action {action} is higer than available 0 up to {self.n_processes-1}'

        binary_action = self.binary_single_action_by_actor(action=action)
        attention_idx = np.where(binary_action)[0]
        for at in attention_idx:
            belief = self.process_belief[i_belief][-1].copy()
            belief_tmp = belief.copy()
            attention = self.processes[i_belief].iloc[self.time[i_belief], at]
            belief_pi_t = np.zeros(shape=self.n_processes)
            f_t = self.get_probability_density(attention, self.sigma, 0)
            g_t = self.get_probability_density(attention, self.sigma, 1)
            for j in range(self.n_processes):
                if j != at:
                    num = belief_tmp[j] * f_t
                else:
                    num = belief_tmp[j] * g_t
                denom = (1 - belief_tmp[at]) * f_t + belief_tmp[at] * g_t
                belief_pi_t[j] = num / denom
            self.process_belief[i_belief].append(belief_pi_t)
        self.previous_actions_count = np.sum(binary_action)
        self.actions_count[i_belief] += self.previous_actions_count
        self.belief[i_belief].append(self.process_belief[i_belief][-1])
        return 0


    def binary_single_action_by_actor(self, action):
        return list(self.available_hypothesis.iloc[action])

    @staticmethod
    def get_probability_density(x, sigma, mu):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

    def gather_pref(self,agent):
        true_hyp=self.individual_hypotheses.name
        self.hypes_successes[agent][str(true_hyp)].append(int(true_hyp==np.argmax(self.belief[agent][-1])))
        self.hypes_lengths[agent][str(true_hyp)].append(self.time[agent])
        self.good_episodes[agent]+=1

    def render(self, mode='training', save_path=None,episode_number=None,agent=None):
        fig, ax = plt.subplots(3,1,figsize=(12,9))
        fig.suptitle('Time-Steps',y=0.05, fontsize=16)
        fig.subplots_adjust(hspace=0.25)
        xdata = []
        ydata=[[] for _ in range(self.n_processes+1)]
        bar_data=[]
        ln = [ax[0].plot([], []) for _ in range(self.n_processes+1)]
        colormap = plt.cm.rainbow
        colors = [list(colormap(i)) for i in np.linspace(0, 1, len(ax[0].lines))]
        colors[-1]=[0.0,0.4,0.0,1.0]
        __colors=[matplotlib.colors.rgb2hex(i) for i in colors]
        __linestyles = ['o', '^', 'v', '>', '<', '*', 'h', 'd', 'p', 'H']
        for i, j in enumerate(ax[0].lines):
            j.set_color(colors[i])
            if i<self.n_processes:
                j.set_marker(__linestyles[i])
            elif i == self.n_processes:
                j.set_marker('D')
        ydata_actions = []

        ydata_processes = [[] for _ in range(self.n_processes)]
        ln_processes = [ax[2].plot([], []) for _ in range(self.n_processes)]
        for i in range(self.n_processes):
            ln_processes[i][0]._color=__colors[i]
            ln_processes[i][0].set_marker(__linestyles[i])


        def init():
            ax[0].set_xlim(0, self.time[agent])
            ax[0].set_ylim(0, 1)
            ax[1].set_xlim(0, self.time[agent])
            ax[1].set_ylim(0, self.n_processes+2)
            ax[2].set_xlim(0, self.time[agent])
            ax[2].set_ylim(-7, 7)

            ax[0].set_xlabel('(a)', fontsize=14)
            ax[1].set_xlabel('(b)', fontsize=14)
            ax[2].set_xlabel('(c)', fontsize=14)

            ax[0].set_ylabel('State', fontsize=16)
            ax[1].set_ylabel('Action', fontsize=16)
            ax[2].set_ylabel('Noisy Measurements', fontsize=16)

            return ln, ln_processes,

        def update(frame):
            xdata.append(frame[0])
            bar_data.append(frame[self.n_processes+1])
            for i_plot in range(self.n_processes):
                ydata[i_plot].append(frame[i_plot+1])
                ln[i_plot][0].set_data(xdata, ydata[i_plot])
            ydata_actions.append(frame[self.n_processes+1]+1)
            lines = [[(index_action,action),(index_action+1,action)] for index_action,action in enumerate(ydata_actions)]
            lc = collections.LineCollection(lines,colors=[colors[c-1] for c in ydata_actions], linewidths=2)
            ax[1].add_collection(lc)
            if ydata_actions[-1] == self.n_processes + 1:
                ax[1].scatter(len(ydata_actions) - 0.5, ydata_actions[-1], marker='D', c=__colors[ydata_actions[-1] - 1])
            elif ydata_actions[-1] < self.n_processes + 1:
                ax[1].scatter(len(ydata_actions) - 0.5, ydata_actions[-1], marker=__linestyles[ydata_actions[-1] - 1], c=__colors[ydata_actions[-1] - 1])

            for j_plot in range(self.n_processes):
                ydata_processes[j_plot].append(frame[self.n_processes+j_plot+2])
                ln_processes[j_plot][0].set_data(xdata, ydata_processes[j_plot])
            ydata[self.n_processes].append(frame[-1])
            return ln, ln_processes,


        ani = FuncAnimation(fig, update,
                            frames=np.concatenate((np.linspace(0,len(self.belief[agent][:-1])-1,len(self.belief[agent][:-1]),dtype=np.int).reshape((self.time[agent],1)),
                                                   np.array(self.belief[agent][:-1]),
                                                   np.array(self.actions_taken[agent]).reshape((self.time[agent],1)),
                                                   np.array(self.processes[agent])),
                                                   axis=1),
                            init_func=init, blit=False,interval=2,repeat=False,cache_frame_data=False)
        fig.legend(list(self.individual_hypotheses.index) + ['Terminate+'],loc='center right', bbox_to_anchor=(1.01,0.5), fontsize=10)

        if save_path != None:
            if not os.path.isdir(os.path.join(save_path, 'render')):
                os.makedirs(os.path.join(os.path.join(save_path, 'render')))
            writegif = matplotlib.animation.PillowWriter(fps=10)
            if self.time[agent]<self.horizon_punishment:
                ani.save(os.path.join(save_path, 'render', f'independent_agents_process_{self.individual_hypotheses.name+1}_episode_{episode_number}_agent_{agent}.gif'), writer=writegif)
            else:
                ani.save(os.path.join(save_path, 'render', f'independent_agents_process_{self.individual_hypotheses.name+1}_episode_{episode_number}_agent_{agent}_NE.gif'), writer=writegif)
            plt.ion()
            if self.time[agent] < self.horizon_punishment:
                fig.savefig(os.path.join(save_path, 'render', f'independent_agents_process_{self.individual_hypotheses.name+1}_episode_{episode_number}_agent_{agent}.png'))
            else:
                fig.savefig(os.path.join(save_path, 'render', f'independent_agents_process_{self.individual_hypotheses.name+1}_episode_{episode_number}_agent_{agent}_NE.png'))

        else:
            plt.ion()
            plt.pause(0.1)

        plt.close()
        return colors