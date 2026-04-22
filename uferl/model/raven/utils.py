def parse_obs(obs_shape):
    camera_names = []
    gripper_names = []
    gravity = False
    for k in obs_shape.keys():
        if k.endswith('_image'):
            camera_names.append(k.rsplit('_image', 1)[0])
        elif k.endswith('_eef_pos'):
            gripper_names.append(k.rsplit('_eef_pos', 1)[0])
        elif k == 'gravity':
            gravity = True

    return camera_names, gripper_names, gravity
