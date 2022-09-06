import os
import cv2
import datetime
import numpy as np
import pickle
import argparse

import nav.environments as environments
import nav.interfaces as interfaces

from path import *

## Collect demo data
def demo(task, env_type_id):

    # name the episode
    episode = datetime.datetime.now().strftime("%m%d_%H%M%S")

    # init env, buffers
    dic_record_array = {}
    dic_record_array['color'] = []
    dic_record_array['depth'] = []
    dic_record_array['vel_x'] = []
    dic_record_array['yaw_rate'] = []
    dic_record_array['yaw'] = []
    dic_record_array['bird_view'] = []

    dic_data_array = {}
    dic_data_array['observation'] = []
    dic_data_array['action'] = []
    dic_data_array['yaw'] = []
    dic_data_array['done'] = []

    space_mouse = interfaces.SpaceMouse()

    env = environments.build_demo_env(render=True, env_type = env_type_id)
    env.reset()
    done = False

    action = np.zeros(2)

    # Simulatio loop
    while not done:
        
        new_input =  space_mouse.control
        action = 0.7 * action + 0.3 * np.array([np.clip(new_input[0]+0.7, 0, 1.0), np.clip(new_input[-1], -1.0, 1.0)])
        obs, rew, done, info = env.step(action)

        dic_data_array['observation'].append(obs['rgbd'])
        dic_data_array['yaw'].append(obs['yaw'])
        dic_data_array['action'].append(action)
        dic_data_array['done'].append(done)

        dic_record_array['color'].append(env.render('rgb'))
        dic_record_array['depth'].append(env.render('depth'))
        dic_record_array['vel_x'].append(action[0])
        dic_record_array['yaw_rate'].append(action[1])
        dic_record_array['yaw'].append(obs['yaw'])
        dic_record_array['bird_view'].append(env.render('fpv'))

    if info['Fail']:
        episode += '_fail'

    # Save data
    os.makedirs(os.path.join(PATH_DATASETSET_SIM, task), exist_ok=True)
    with open('{}/{}/{}.pickle'.format(PATH_DATASETSET_SIM, task, episode), 'wb') as handle:
        pickle.dump(dic_data_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(os.path.join(PATH_RAW_SIM, task, episode), exist_ok=True)
    with open('{}/{}/{}/data.pickle'.format(PATH_RAW_SIM, task, episode), 'wb') as handle:
        pickle.dump(dic_record_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    video_format = cv2.VideoWriter_fourcc(*'mp4v')
    for key in ['color', 'depth', 'bird_view']:

        if key == 'bird_view':
            recorder = cv2.VideoWriter("{}/{}/{}/{}.mp4".format(PATH_RAW_SIM, task, episode, key), video_format, 20, (480, 360))
        else:
            recorder = cv2.VideoWriter("{}/{}/{}/{}.mp4".format(PATH_RAW_SIM, task, episode, key), video_format, 20, (212, 120))
        for img in dic_record_array[key]:
            if key == 'depth':
                recorder.write(img[:, :, np.newaxis][:,:,[0,0,0]])
            else:
                recorder.write(img)
        recorder.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type",type=int,default=2,
                        help="2-Easy, 0,1-Medium, 1,4: Hard")
    parser.add_argument("--demo_name",type=str,default='corl2022',
                        help="name of directory to save demo data.")
    args = parser.parse_args()
    assert(args.env_type in range(5))

    demo(args.demo_name+str(args.env_type), args.env_type)
