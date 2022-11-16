# develop programm to generate a real world dataset (not synthesized)

import cv2
from threading import Thread
import time

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


    while True :

        if webcam_stream.stopped is True:
            break
        else :
            frame = webcam_stream.read()

        currtime = str(round(time.time(), 2))
        cv2.imwrite("C://Data/dehaze/CustomData/" + currtime + ".png", frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        time.sleep(60)


    webcam_stream.stop()
    cv2.destroyAllWindows()