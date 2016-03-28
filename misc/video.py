import cv2
import numpy as np
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description="Change color space of the input\
    video stream using keyboard controls: Grayscale - 'g', YUV - 'y', HSV - 'h'")
    return parser

def cartoonize_image(img, ds_factor=4, sketch_mode=False):
    #convert to gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #apply median filter
    img_gray = cv2.medianBlur(img_gray, 7)

    #detect edges and threshold the imag
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    
    #mask is the sketch of the image
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation = cv2.INTER_AREA)
    num_repetitions = 10
    sigma_color = 5
    sigma_space = 7
    size = 5
    
    #apply bilateral filter multiple times
    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, size, sigma_color, sigma_space)
    
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)
    dst = np.zeros(img_gray.shape)
    
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst
    

if __name__ == "__main__":
    
    args = argument_parser().parse_args()

    cap = cv2.VideoCapture(0)

    #check if opened correctly
    if not cap.isOpened():
        raise IOError('Cannot open webcam')

    cur_char = -1
    prev_char = -1
    
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
        c = cv2.waitKey(1)
        #Esc key (ASCII: 27)
        if c == 27:
            break
            
        if c > -1 and c!=prev_char:
            cur_char = c
        prev_char = c
        
        if cur_char == ord('g'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif cur_char == ord('y'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        elif cur_char == ord('h'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
        elif cur_char == ord('s'):
            output = cartoonize_image(frame, sketch_mode=True)
        elif cur_char == ord('c'):
            output = cartoonize_image(frame, sketch_mode=False) 
        else:
            output = frame
        
        cv2.imshow('Webcam', output)

    cap.release()
    cv2.destroyAllWindows()
    