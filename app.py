from guizero import App, PushButton
from DeepLabStream import DeepLabStream, show_stream
from utils.configloader import MULTI_CAM, STREAMS, RECORD_EXP


def start():
    # starting streams
    stream_manager.start_cameras(STREAMS, MULTI_CAM)
    app.repeat(0, main_loop)

    start_button.disable()
    stop_button.enable()
    analysis_start.enable()
    recording_start.enable()
    app.update()


def main_loop():
    all_frames = stream_manager.get_frames()
    color_frames, depth_maps, infra_frames = all_frames

    # writing the video
    # video_frames = deepcopy(color_frames)
    if stream_manager.recording_status():
        stream_manager.write_video(color_frames, stream_manager.frame_index)

    if stream_manager.dlc_status():
        # outputting the frames
        res_frames, res_time = stream_manager.get_analysed_frames()
        # inputting the frames
        # input_frames = (color_frames, depth_maps, infra_frames)
        stream_manager.input_frames_for_analysis(all_frames, stream_manager.frame_index)
        # streaming the stream
        if res_frames:
            show_stream(res_frames)
    else:
        show_stream(color_frames)

    stream_manager.frame_index += 1


def start_analysis():
    print("Analysis starting")
    stream_manager.set_up_multiprocessing()
    stream_manager.start_dlc()
    stream_manager.create_output()

    experiment_start.enable()
    analysis_start.disable()
    analysis_stop.enable()
    app.update()


def stop_analysis():
    print("Analysis stopped")
    stream_manager.stop_dlc()

    experiment_start.disable()
    analysis_start.enable()
    analysis_stop.disable()
    app.update()


def start_experiment():
    print("Experiment started")
    stream_manager.set_up_experiment()
    stream_manager.start_experiment()
    if RECORD_EXP:
        start_recording()

    experiment_start.disable()
    experiment_stop.enable()
    app.update()


def stop_experiment():
    print("Experiment stopped")
    stream_manager.stop_experiment()
    if RECORD_EXP:
        stop_recording()

    experiment_stop.disable()
    experiment_start.enable()
    app.update()


def start_recording():
    print("Recording started")
    stream_manager.start_recording()

    recording_start.disable()
    recording_stop.enable()
    app.update()


def stop_recording():
    print("Recording stopped")
    stream_manager.stop_recording()

    recording_stop.disable()
    recording_start.enable()
    app.update()


def stop():
    print("Should stop here")
    stream_manager.finish_streaming()
    app.cancel(main_loop)

    start_button.enable()
    stop_button.disable()
    analysis_start.disable()
    analysis_stop.disable()
    experiment_start.disable()
    experiment_stop.disable()
    recording_start.disable()
    recording_stop.disable()
    app.update()


if __name__ == "__main__":
    print("Starting DeepLabStream manager")
    stream_manager = DeepLabStream()
    app = App(title="DeepLabStream", width=800)
    start_button = PushButton(app, command=start, text="Start Streaming", align="top", width=30, height=5)
    stop_button = PushButton(app, command=stop, text="Stop Streaming", align="bottom", width=30, height=5,
                             enabled=False)
    analysis_start = PushButton(app, command=start_analysis, text="Start Analysis", align="left", width=15,
                                height=5, enabled=False)
    experiment_start = PushButton(app, command=start_experiment, text="Start Experiment", align="left", width=15,
                                  height=5, enabled=False)
    recording_start = PushButton(app, command=start_recording, text="Start Recording", align="left", width=15, height=5,
                                 enabled=False)
    analysis_stop = PushButton(app, command=stop_analysis, text="Stop Analysis", align="right", width=15, height=5,
                               enabled=False)
    experiment_stop = PushButton(app, command=stop_experiment, text="Stop Experiment", align="right", width=15,
                                 height=5, enabled=False)
    recording_stop = PushButton(app, command=stop_recording, text="Stop Recording", align="right", width=15, height=5,
                                enabled=False)
    app.display()
