import deeplake
import pandas as pd
import os
import skvideo.io
import numpy as np
import tqdm


def downsample_videos(source_dir, target_dir, scale, w0 = 424, h0 = 264, num_train_scenes = 288):
    ## To load datasets from the hosted servers we have to load an API key, which you can create under your account, and store it as an environment variable. -->
    # Resizing Videos
    w = int(scale * w0)
    h = int(scale * h0)

    with tqdm(desc='Rescaling training scenes', total=num_train_scenes) as prog_bar:
    #     # TODO: consider using Hmov.ScenePars.load_scene() instead    
        for i in range(num_train_scenes):
            
            # naming convention for berenslab machines
            file_name_in = f'hmovTrain_v3_{w0}x{h0}_scene-{i + 1:03d}.avi'
            file_name_out = f'hmovTrain_v3_{w}x{h}_scene-{i + 1:03d}.avi'
            filepath_in = os.path.join(source_dir, file_name_in)
            filepath_out = os.path.join(target_dir, file_name_out)
            
#           # Load video frames
            downscaled_vid = skvideo.io.vread(filepath_in,
                                            as_grey=True, # False by default, which would then load as RGB, with 3 identical channels
                                            outputdict={
                                                "-sws_flags": "bilinear",
                                                "-s": f'{w}x{h}'
                                            }
                                            ) # downscaled_vid has three channels but all are identical
            
            # changing framerate to 60 Hz to match PSTH!
            skvideo.io.vwrite(filepath_out, downscaled_vid, inputdict={'-r': '60'}, outputdict={"-pix_fmt": "gray"}) # {'-r':'30'} denotes frame(r)ate
            prog_bar.update(1)


def create_deeplake(dlgn_ds_path, resp_file_path, scene_col, id_col, target_col, video_dir, w, h):


    dlgn_ds = deeplake.empty(dlgn_ds_path, overwrite=True)
    with dlgn_ds:
        dlgn_ds.create_tensor('id',dtype='str')
        dlgn_ds.create_tensor('videos', dtype='float32')
        dlgn_ds.create_tensor('responses', dtype='float32')

    resp_df = pd.read_pickle(resp_file_path)
    resp_pivoted = resp_df.pivot_table(
        index=scene_col,
        columns=id_col,
        value=target_col
    ) # organize response by stimulus; each row is a scene, each column is a different detected unit, and each item in table is entire psth of that neuron in response to that scene

    # iterate through scenes and append responses/videos to deeplake
    with dlgn_ds:
        # dlgn_ds.videos.extend(  train_rescaled  ) # .extend treats first axis as sample index, which is desired in this case
        for ind, row in resp_pivoted.iterrows():
            
            vid_filename = f'hmovTrain_v3_{w}x{h}_scene-{ind+1:03d}.avi'
            vid_filepath = os.path.join(video_dir, vid_filename)
            vid_arr = skvideo.io.vread(vid_filepath,as_grey=True, ).transpose(3,0,1,2) # read in video, already downscaled
            # append in (c,t,h,w) order, thus transpose
            # np.stack(vid,np.tile(behavior,...)) # tile and append behavior as new channel
            # extract optogenetic signal for scene, tile to size of frame
            # vid = np.stack((vid,opto_array)).squeeze()
            dlgn_ds.append({'responses': np.stack(row.values).astype(np.float32),
                            'videos': vid_arr.astype(np.float32),
                            'id': str(ind)
                        }, skip_ok=True)