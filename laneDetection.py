import cv2
import numpy as np
import matplotlib.pyplot as plt


# loading the image
# image= cv2.imread('test_image.jpg')
# cv2.imshow('result', image)

# canny edge detection: identify sharp change in intesity
# gradient: change in brightness over adjacent area
# edge:rapid change in brightness

# GRAY CONVERSION
# lane_image= np.copy(image)
# gray= cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY) #CANNY conversion autoatically applies 5*5  gaussian blur

# reduce noise:
# blur = cv2.GaussianBlur(gray, (5,5),0)

# identifying edges (identifying steep change) i.e. canny edge detection
# areaes where complete black corresponds to low intensity changes b/w adjacent pixels. whereas the white line represents a region of high gradient.
# i.e change in intensity exceeding the threshold.
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parametrs = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parametrs[0]
        intercept = parametrs[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
    # print(left_fit_average, "left")
    # print(right_fit_average, "right")


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(image):
    height = image.shape[
        0]  # shape of an array is denoted by tuple of  integers e.g. height= image.shape[m,n,l] where m=height, n=width, l=depth
    polygon = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)  # area filled with triangle will  be comletely white.
    masked_image = cv2.bitwise_and(image, mask)  # taking and of mask with image
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


# finding lane line lines (hough transform, identifying straight line)
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

# finding lane lines
# threshold: minimum no of intersection (votes) needed to accept a candidate line
# line in coordinate plane =  single pointin huff space.
# family of lines in huff space = points corresponding to particular staight line.
# identifying possible lines from series of points.
# arguments: 1st- image, 2nd-  pixels to cheeck at a time, 3rd- angle of check, 4th- minimum no. of intersection, 5th- empty array for storing
# the processed image, 6th- minimum line length, minimum line gap
 # Since matplotlib also contains imshow and show(no need to write waitkey, just writ show) fxn and here no need to  give the window name
# plt.imshow(canny)
# plt.show()

cap = cv2.VideoCapture("inputVideo.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(30)== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

