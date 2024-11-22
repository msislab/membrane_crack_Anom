import cv2


class GetROI(object):
    def __init__(self) -> None:
        self.roi = []
    
    def _getROI(self, img=None, windowName=None):
        self.roi = []
        clone = img.copy()
        cv2.namedWindow(f"{windowName}")
        cv2.setMouseCallback(f"{windowName}", self.click_and_crop)
        
        while True:
            cv2.imshow(f"{windowName}", img)
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                img = clone.copy()
            # if the 'c' key is pressed, break from the loop
            elif len(self.roi) == 2:
                cv2.destroyAllWindows()
                break
        return self.roi

    def click_and_crop(self, event, x, y, flags, param):

        # if event == cv2.EVENT_LBUTTONDOWN:
        #     _refPt = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self.roi.append((x,y))

if __name__=="__main__":
    GetROI()      