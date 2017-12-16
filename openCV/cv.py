import cv2
import numpy
import matplotlib.pyplot

def main():
    img = cv2.imread('opencv-logo2.png')
    print img
    print img[100][100]
    blue = img[100, 100, 0]
    print "blue before={}".format(blue)
    img[100, 100] = [0, 0, 0]
    print "blue after={}".format(img[100, 100, 0])
    red = img.item(10, 10, 2)
    print "red before= {} ".format(red)
    img.itemset((10, 10, 2), 100)
    print "Red after = {}".format(img[10, 10, 2])
    print "shape = {}".format(img.shape)
    print "size = {}".format(img.size)
    print "dtype = {}".format(img.dtype)
    ball = img[250:450, 250:450]
    img[0:200, 0:200] = ball
    cv2.imwrite('new_image.png', img)
    b, g, r = cv2.split(img)
    print "blue:{}".format(b)
    print "red:{}".format(g)
    print "green:{}".format(r)
    cv2.imwrite('blue.png', b)
    cv2.imwrite('red.png', r)
    cv2.imwrite('green.png', g)
    img = cv2.merge((b, g, r))
    cv2.imwrite('merged.png', img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()