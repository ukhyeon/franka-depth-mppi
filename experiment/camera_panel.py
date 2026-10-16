"""Local RGB/depth display through Isaac Sim UI; no OpenCV HighGUI or image saving."""
import numpy as np
import cv2
import omni.ui as ui
class LiveCameraPanel:
    def __init__(self):
        self.stop_requested=False;self.frames=0
        self.provider=ui.ByteImageProvider()
        self.window=ui.Window('D435i RGB + depth',width=680,height=330)
        with self.window.frame:
            with ui.VStack():
                ui.Label('RGB left | depth right (0-0.7m) | unmirrored optical view')
                ui.ImageWithProvider(self.provider,height=240)
                self.status=ui.Label('Waiting for camera frame')
                ui.Button('Stop demo',clicked_fn=self.stop)
    def stop(self):self.stop_requested=True
    def update(self,data):
        rgb=data['raw_rgb'];depth=data['raw_depth']
        valid=depth>0
        shade=np.clip(depth*(255/.7),0,255).astype(np.uint8)
        colored=cv2.applyColorMap(shade,cv2.COLORMAP_VIRIDIS);colored[~valid]=0
        bgr=np.hstack([cv2.resize(rgb,(320,240)),cv2.resize(colored,(320,240))])
        rgba=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGBA)
        self.provider.set_bytes_data(rgba.ravel().tolist(),[640,240])
        self.frames+=1
        self.status.text='Frame %d | valid depth %.1f%%'%(self.frames,100*valid.mean())
        if self.frames==1:print('CAMERA_PANEL_FRAME',{'source_shape':list(depth.shape),'display_rgba_shape':list(rgba.shape),'valid_depth_fraction':float(valid.mean()),'depth_unit':'m'},flush=True)
