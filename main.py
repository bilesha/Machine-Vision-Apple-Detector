import cv2 as cv
import numpy as np
from absl import app
from pathlib import Path

def main(argv):

    executable_path = Path(argv[0])

    input_file = (executable_path.parent).joinpath("0.png")

    print(input_file)

    image = cv.imread(str(input_file), cv.IMREAD_COLOR)

    if image is None:

        print("Error: 'image' is empty")

        return 1

    cv.imshow("image", image)

    #resize the original image
    width , height = 640, 480
    imageResize = cv.resize(image, (width, height))
    print(imageResize.shape)
    cv.imshow("resized image", imageResize)

    blurredimage = cv.GaussianBlur(imageResize,(5,5),0)
    cv.imshow("BluredImage", blurredimage)

    r = 480/ width 
    dim = (480, int(height*r))
    AspectRatioImage =  cv.resize(blurredimage, dim)
    cv.imshow("Aspect ratio ", AspectRatioImage)

    mediumblurredimage = cv.medianBlur(AspectRatioImage, 5)
    cv.imshow("MEdium Blured Image", mediumblurredimage)

    grayimage = cv.cvtColor(mediumblurredimage, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray Image", grayimage)

    #light 
    lightresizedimage = cv.cvtColor(mediumblurredimage,cv.COLOR_BGR2HSV)
    equalizelight = (cv.split(lightresizedimage))[2]
    cv.merge(cv.split(lightresizedimage),lightresizedimage)
    cv.imshow("Light filtered image",lightresizedimage)
    cv.waitKey(0)

    newfinalImage = cv.cvtColor(lightresizedimage,cv.COLOR_HSV2BGR) 
    cv.imshow("Fused final image",newfinalImage)
    cv.waitKey(0)

    #Colour change image using resized image
    colourimage = cv.cvtColor(newfinalImage,cv.COLOR_BGR2HSV)

    colourchangeimage = colourimage [:,:,0] 
    cv.imshow("Colour hue change", colourchangeimage)
    cv.waitKey(0)

    #red 
    RedImage = cv.inRange(colourimage,(0,0,0),(70,255,255))
    cv.imshow("Red image",RedImage)
    cv.waitKey(0)

    #green 
    GreenImage = cv.inRange(colourimage,(160,0,0),(180,255,255))
    cv.imshow("Green Image",GreenImage)
    cv.waitKey(0)

    RedandGreenImage = RedImage + GreenImage
    cv.imshow("Red and Green Image",RedandGreenImage)

    #errosion and dilution for loop
    shape = cv.MORPH_ELLIPSE
    kernel = cv.getStructuringElement(shape,(3,3))
    dilatedimage = cv.dilate(RedandGreenImage,kernel,iterations = 2)
    erodedimage = cv.erode(dilatedimage,kernel,iterations = 3)
    dilatedimage2 = cv.dilate(erodedimage,kernel,iterations = 3)
    erodedimage2 = cv.erode(dilatedimage2,kernel,iterations = 5)
    cv.imshow("Eroded and Dilated image",erodedimage2)

    applecount = 0 
    drawingCircleImage = cv.cvtColor(erodedimage2, cv.COLOR_BAYER_BG2GRAY)
    contourcircle = cv.HoughCircles(drawingCircleImage, cv.HOUGH_GRADIENT, 1, 20, param1 = 150, param2 = 26, minRadius = 0, maxRadius = 0)
    count = np.uint16(np.around(contourcircle))
    for (x,y,r) in count[0, :]:
        if r <= 60:
            cv.circle(AspectRatioImage, (x,y), r, (0, 255, 0), 3)
            applecount = applecount + 1
        else:
            pass
    cv.imshow("Output", AspectRatioImage)
    print(applecount, "apples were detected in this image")

    cv.waitKey(0)

    cv.destroyAllWindows()

    return 0


if __name__ == "__main__":

    app.run(main)