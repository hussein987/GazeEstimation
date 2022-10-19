import cv2
import math
import numpy as np

def get_eye_pupil(img_file):
    img = cv2.imread(img_file)
    scaling_factor = 0.5


    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', img)
    gray = cv2.cvtColor(~img, cv2.COLOR_BGR2GRAY)

    ret, thresh_gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        rect = cv2.boundingRect(contour)
        x, y, width, height = rect
        radius = 0.25 * (width + height)
        print(img.shape, width, height, area)

        area_condition = (100 <= area <= 225)
        symmetry_condition = (abs(1 - float(width)/float(height)) <= 0.2)
        fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= 0.3)

        print("Ckecking...")
        if area_condition and symmetry_condition and fill_condition:
            print("HERE")
            print(int(x + radius), int(y + radius))
            cv2.circle(img, (int(x + radius), int(y + radius)), int(radius), color=(255, 0, 0))
            cv2.circle(img, (int(x + radius), int(y + radius)), 0, color=(0, 0, 255))

    cv2.namedWindow('Pupil Detector', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Pupil Detector', img)

    c = cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    get_eye_pupil('dataset/UnityEyes/imgs/1.jpg')


if __name__ == "__main__":
    main()