# %matplotlib inline
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from IPython.display import HTML, display
import numpy as np
import ipdb

# model.core.?? layer? ??.parameters
# SVD(filters in first layer, take 1 for space and 1 for time I guess??

# 

def plot_raster(df, stim_index):
    # create subplots per session if unique id

    fig, ax = plt.subplots()

    for unit, times in df['spike_times'].iloc[stim_index].iterrows(): # take arb. row ind., list of spike times
        ax.vlines(times, unit - 0.5,unit + 0.5, ) # plot vertical bars
        
        # ax.set_xlim([0, len(df[''])])
    ax.set_xlabel('Time (ms)')

    ax.set_yticks(len(df['spike_times']))
    ax.set_ylabel('Trial Number')

    ax.set_title(f'Spike Times, Video {stim_index}')

    '''
    # add shading for stimulus duration)
    ax.axvspan(light_onset_time, light_offset_time, alpha=0.5, color='greenyellow')
    '''

    return fig, ax # how to return?
        


# def plot_psth(psth, stim_index):
#     # create subplots per session if unique id
#     for series in psth.iloc[stim_index]


def playback_vid(vid, index, fs=60):
    '''
    :param vid: Deeplake/Torch Tensor with shape (c,h,w,t)
    :param index: Deeplake.Index entry ( or another string which contains numeric index)
    :param fs: sampling rate (framerate of video) 
    '''
    video = vid.numpy().transpose(1,2,3,0) # must be in shape (t, h, w, c)
    
    fig, ax = plt.subplots()
    im = plt.imshow(video[0,:,:,:])
    
    plt.close() # this is required to not display the generated image
    
    def init():
        im.set_data(video[0,:,:,:])
    
    def animate(i):
        im.set_data(video[i,:,:,:])
        return im
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0],
                                   interval=1000/fs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set(title = f"Scene #{''.join([x for x in str(index) if x.isdigit()]) }" )
    
    display(HTML(anim.to_html5_video()))



def batch_playback(batch, neur_ID):
    B = len(batch['videos'])
    for b in range(B):
        batchlet = {k: v[b,::] for k, v in batch.items()} # take one sample, remove batch dim
        multi_playback(batchlet, neur_ID)


def multi_playback(batchlet, neur_ID, fs=60, neur_ind=None):
    '''Displays concurrent video, response, behavioral measurements playback
    :param batchlet: batch of data, for now with batch size 1 (first dim has len 1).  
        Tensors should have shape (c,t,[x,y]), transpose to (t,[x,y],c)
    :param neur_ID: 
    :param neur_ind: for if only certain neurons from the batch should be displayed

    '''
    vid, resp, opto = batchlet['videos'].numpy().transpose(1,2,3,0), batchlet['responses'].numpy(), batchlet['opto'].numpy()
    N, T = len(resp), len(opto) # number of neurons
    t = np.arange(0,T) / fs
    
    # Create the figure and grid for subplots
    fig = plt.figure(figsize=(15, 8)) # we want video, pupil_center on left, line plots on right, neurons on right most? (or above/below other line plots)
    gs = GridSpec(N + 1, 3, figure=fig)  # N+1 rows for N neurons plus opto, 2 columns for vid vs. line plots

    vid_ax = fig.add_subplot(gs[:,0])
    opto_ax = fig.add_subplot(gs[0,1:])
    neur_ax = [ fig.add_subplot(gs[n+1, 1:] , ) for n in range(N) ]

    resp_lines = [ax.plot([],[])[0] for ax in neur_ax] # all of these [0]'s are because ax.plot returns list of Line2D objects
    opto_line = opto_ax.plot([],[])[0]
    vid_disp = vid_ax.imshow(vid[0,:,:,:],cmap='gray')
    
    ylim = resp.max()
    for i, ax in enumerate(neur_ax):
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim([-0.05, t[-1] + 0.1 ])
        ax.set_ylim([-0.1, ylim + 0.1])
        ax.set_title(f'Unit #{neur_ID[i][-1]}', fontsize=8)

    opto_ax.set_xlim([-0.05, t[-1] + 0.1 ])
    opto_ax.set_ylim([-0.1, 1.1 ])
    opto_ax.set_yticks([0,1])
    opto_ax.set_yticklabels(['Off','On'])
    opto_ax.set_title('Optogenetic Signal',fontsize=8)
    
    vid_ax.set_xticks([])
    vid_ax.set_yticks([])
    vid_ax.set_title('Stimulus Shown')

    fig.suptitle(f"Scene #{''.join([x for x in str(batchlet['index']) if x.isdigit()]) } \n Mouse {neur_ID[0][0]}, Session {neur_ID[0][1]}, Experiment {neur_ID[0][2]}, " )

    # # Function to initialize the line plot subplot
    # def init_vid(video):
    #     return vid_ax.imshow(video[0,:,:,:])

    # def init_opto(): # line is
         

    # def init_resp(neur_ax): # line is
    #     [ ax.plot([], []) for ax in neur_ax ] 

    # def init_all():
    #     vid_art = init_vid(vid)
    #     opto_art = init_opt()
    #     resp_arts = init_resp()
    #     vid_art, opto_art, resp_arts
    
    # Function to update the line plot subplot
    def update_resp(frame):
        for line, neur in zip(resp_lines, resp):
            line.set_data(t[:frame], neur[:frame])
        return resp_lines

    def update_opto(frame):
        opto_line.set_data(t[:frame], opto[:frame])
        return opto_line
    
    def update_vid(frame):
        vid_disp.set_data(vid[frame,:,:,:])
        return vid_disp

    def animate(frame):
        # if blit=True, must return sequence of mpl.artist.Artist's, so cast singletons to lists and concatenate
        return update_resp(frame) +  [ update_opto(frame) ] + [ update_vid(frame) ] 
    
    plt.tight_layout()
    # Create the line plot animation
    anim = animation.FuncAnimation(fig, animate, frames=T, interval = 1000/fs, blit=True)
    # ipdb.set_trace()
    # plt.tight_layout()
    display(HTML(anim.to_html5_video()))
    




# def show_readout_locs(model, neuron_ind):
    
#     for readout in model.readouts:
#         readout.sample_grid()

    
#     return readout_mu, readout_sigma

    

def plot_obs_pred_psth(stim_ind, neur_ind, resp_array, model_pred, neur_ID, T = 5, Hz = 60):
    # plot trace of example neuron and stimulus 
    # will just feed in result of forward pass on dataset, which should be of same size as dataset.responses
    # as this is, stim_ind is not the scene ID of the video, but just the literal ordinal index of a scene in the dataset
    # (b traces, t time, c neurons)
    
    # model_pred: (b,c,t)
    # resp_array is being fed in (b,c,t)!
    
    
    psth_obs = resp_array #[stim_ind,neur_ind] # (len(neur_ind),300)
    psth_pred = model_pred #[stim_ind][:,neur_ind] # not sure if there's a more syntactically uniform way to achieve this (like, identically for the two arrays)
    # neur_ID = [ dataset.info.neuron_ids[ind] for ind in neur_ind ]
    t_ax = np.linspace(0,T,int(T*Hz))
    
    f, ax = plt.subplots(len(neur_ind),len(stim_ind), figsize = (len(stim_ind)*4,len(neur_ind)*2) )
    for i, n in enumerate(neur_ind):
        for j, s in enumerate(stim_ind):
            ax[i,j].plot(t_ax,psth_obs[j,i],'g',label='True')
            ax[i,j].plot(t_ax,psth_pred[j,i],label='Predicted')
            ax[i,j].set(title=f' Pred vs. Obs PSTH \n Unit {neur_ID[i]}, Stim {s}',
                        xlabel = 'Time (s)',
                        ylabel = 'Spikes/sec?',
                        xticks = [],
                        yticks = [])
            ax[i,j].legend()
            ax[i,j].title.set(fontsize=7)
            ax[i,j].xaxis.label.set(fontsize=6)
            ax[i,j].yaxis.label.set(fontsize=6)
    plt.tight_layout()

    return f