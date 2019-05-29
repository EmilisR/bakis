import cv2
import numpy as np
import glob
import pytesseract
import time
import sys

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshGauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 27)
    ratio = 200.0 / image.shape[1]
    dim = (200, int(image.shape[0] * ratio))
    resizedCubic = cv2.resize(threshGauss, dim, interpolation=cv2.INTER_CUBIC)
    bordersize = 10
    border = cv2.copyMakeBorder(resizedCubic, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    edges = cv2.Canny(border, 50, 150, apertureSize=3)
    cv2.imshow('d', border)
    cv2.waitKey(1000)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]), minLineLength=100, maxLineGap=80)
    a, b, c = lines.shape
    for i in range(a):
        x = lines[i][0][0] - lines[i][0][2]
        y = lines[i][0][1] - lines[i][0][3]
        if x != 0:
            if abs(y / x) < 1:
                cv2.line(border, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 1, cv2.LINE_AA)

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        gray = cv2.morphologyEx(border, cv2.MORPH_CLOSE, se)
         
    # OCR
    config = ''
    text = pytesseract.image_to_string(gray, lang='training', config=config)
    validChars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(text)
    cleanText = []

    for char in text:
        if char in validChars:
            cleanText.append(char)

    plate = ''.join(cleanText)
    return plate


start = time.time()
plate = detect(cv2.imread(sys.argv[1]))
print(plate)
print(round(time.time() - start, 2), 'ms')

