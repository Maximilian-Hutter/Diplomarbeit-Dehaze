# using https://github.com/JetsonHacksNano/CSI-Camera/blob/master/simple_camera.py and https://gvasu.medium.com/faster-real-time-video-processing-using-multi-threading-in-python-8902589e1055
import cv2
import multiprocessing as mp
import time



def getframe(q, event,fps):
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
        exit()

    frametime =  round(1 / fps, 5)
    while True:
        status, frame = capture.read()
        if status:
            q.put(frame)
        time.sleep(frametime)

        if event.is_set():
            capture.release()
            break

if __name__ == '__main__':

    fps = 60
    q = mp.Queue()
    event = mp.Event()
    proc = mp.Process(target=getframe, args=(q,event,fps,))
    proc.start()
    while True:
        frame = q.get()
        #outframe = Dehaze(frameinfo[1])
        outframe = frame
        cv2.imshow('frame', outframe)
        if cv2.waitKey(1) == ord('q'):
            break

    event.set()
    proc.join()
    cv2.destroyAllWindows()