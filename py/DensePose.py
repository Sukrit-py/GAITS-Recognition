import cv2
import time

def main():
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)  # Add a short sleep before showing the window

    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("unable to read frames")
            break


        cv2.imshow("view", frame)

        key =  cv2.waitKey(1)

        if key == ord('q') or cv2.getWindowProperty("WebCam", cv2.WND_PROP_VISIBLE) < 1:
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "main":
    main()