from picamera2 import Picamera2
import libcamera

def setup_camera(main: dict, lores: dict, fps: int, tranform : dict =None) -> Picamera2:
    picam2 = Picamera2()
    mode = picam2.sensor_modes[0]
    print(f"Camera mode: {mode}")
    video_config = picam2.create_video_configuration(
        main=main, 
        lores=lores, 
        controls={"FrameRate": fps},
        sensor={'output_size': mode['size'], 'bit_depth':mode['bit_depth']})
    if tranform:
        video_config["transform"] = libcamera.Transform(vflip=tranform["vflip"], hflip=tranform["hflip"])
    picam2.align_configuration(video_config)
    picam2.configure(video_config)
    return picam2

def init_camera_defaults(size_main, size_lores, frame_rate=60) -> Picamera2:
    picam2 = setup_camera(
        main={"format": "RGB888", "size": size_main}, 
        lores={"format": "YUV420", "size": size_lores}, 
        fps=frame_rate)
    picam2.set_controls({"AeEnable": True})
    return picam2