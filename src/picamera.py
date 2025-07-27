from picamera2 import Picamera2, Preview
#import libcamera

def setup_camera(main, lores, fps):
    picam2 = Picamera2()
    mode = picam2.sensor_modes[0]
    print(f"Camera mode: {mode}")
    video_config = picam2.create_video_configuration(
        main=main, 
        lores=lores, 
        controls={"FrameRate": fps},
        sensor={'output_size': mode['size'], 'bit_depth':mode['bit_depth']})
        #display="lores" if lores else "main")
    #video_config["transform"] = libcamera.Transform(vflip=1)
    picam2.align_configuration(video_config)
    picam2.configure(video_config)
    #picam2.start_preview(Preview.DRM)
    #picam2.start_preview(Preview.QTGL)
    return picam2