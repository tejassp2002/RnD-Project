import argparse
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import gym
import gym_ple
import highway_env
from fgdqn import FGDQN
from gym.wrappers import Monitor

def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)

    obs = eval_env.reset()
    reward_total = 0 
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()
        reward_total+=reward
    print("Reward ",reward_total)

    # Close the video recorder
    eval_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Catcher-v0',help='Environment Name')
    parser.add_argument('--is_highway', type=str, default='no',help='If using Highway environment pass yes else no') 
    parser.add_argument('--trained_model', type=str,help='Path to trained model')
    parser.add_argument('--n_videos', type=int, default=10,help='Number of videos to record')
    parser.add_argument('--video_length', type=int, default=500, help='Length of each video')
    parser.add_argument('--video_folder', type=str, default='./videos/', help='Folder to save videos')
    
    args = parser.parse_args()
    

    if args.is_highway == 'yes':
        env = gym.make(args.env)
        model = FGDQN.load(args.trained_model,env=env)
        env = Monitor(env, directory=args.video_folder, video_callable=lambda e: True)
        env.set_monitor(env)
        env.configure({"simulation_frequency": 30})  # Higher FPS for rendering

        for videos in range(10):
            done = False
            obs = env.reset()
            i = 0
            while (not done) or (i < args.video_length):
                # Predict
                action, _states = model.predict(obs, deterministic=True)
                # Get reward
                obs, reward, done, info = env.step(action)
                # Render
                env.render()
                i += 1
        env.close()

    else:
        env = DummyVecEnv([lambda: gym.make(args.env)])
        obs = env.reset()
        model = FGDQN.load(args.trained_model,env=env)
        for i in range(args.n_videos):
            record_video(args.env, model, video_length=args.video_length, prefix=f'{args.env}-fgdqn{i}', video_folder=args.video_folder)

