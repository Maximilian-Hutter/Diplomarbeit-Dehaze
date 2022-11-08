# using https://github.com/JetsonHacksNano/CSI-Camera/blob/master/simple_camera.py and https://gvasu.medium.com/faster-real-time-video-processing-using-multi-threading-in-python-8902589e1055
import cv2
from threading import Thread
# from pycoral.adapters import common
# from pycoral.utils.dataset import read_label_file
# from pycoral.utils.edgetpu import make_interpreter

class ImageProcessing:
    def __init__(self, model_path = None):
        #self.interpreter = make_interpreter(model_path)
        model_path = None

    # method to create dehazed frame
    def inference(self):
        return self.frame

    # method to make frame to TF lite compatible
    def transforms(self):

        return self.frame

    # method to output the processed frame
    def process(self, frame):

        return self.inference(self.transforms(frame))


class CamStream:
    def __init__(self, stream_id=0):

        self.stream_id = stream_id
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is initialized to False 
        self.stopped = True
        # thread instantiation  
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads run in background 
        
    # method to start thread 
    def start(self):
        self.stopped = False
        self.t.start()

    # method passed to thread to read next available frame  
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()

    # method to return latest read frame 
    def read(self):
        return self.frame

    # method to stop reading frames 
    def stop(self):
        self.stopped = True

if __name__ == "__main__":
    webcam_stream = CamStream(stream_id=0) # 0 id for main camera
    webcam_stream.start()
    image_processing = ImageProcessing()

    while True :
        if webcam_stream.stopped is True:
            break
        else :
            frame = webcam_stream.read()
        
        frame = image_processing.process(frame)

        cv2.imshow('frame' , frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    webcam_stream.stop()
    cv2.destroyAllWindows()