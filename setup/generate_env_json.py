import os
import json

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    env_json_path = os.path.join(script_path, 'environment.json')
    project_path = os.path.dirname(script_path)

    with open(env_json_path, 'w') as env_json:
        json_dict = {
            "username": os.environ['USER'],
            "train_folder": os.path.join(project_path, 'data', 'train'),
            "validation_folder": os.path.join(project_path, 'data', 'validation'),
            "codebook_path": os.path.join(project_path, 'data', 'codebook.csv'),
            "model_checkpoint_path": os.path.join(project_path, 'checkpoints')
        }
        new_temp_folder = os.path.join(os.path.expanduser('~'), 'temp')
        if not os.path.exists(new_temp_folder):
            os.makedirs(new_temp_folder)
        json_dict['new_temp_folder'] = new_temp_folder

        json.dump(json_dict, env_json)

