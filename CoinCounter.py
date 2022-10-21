
import cv2
import numpy as np
from scipy import ndimage as ndi

def negative_to_zero(img: np.array) -> np.array:
    img = img.copy()
    img[img < 0] = 0
    return img

def convolve(image, kernel) -> np.array:
    k = kernel.shape[0]

    img = np.zeros([image.shape[0], image.shape[1]])
    size = np.array([image.shape[0]-(kernel.shape[0]-1), image.shape[1]-(kernel.shape[0]-1)])
    
    if abs(np.sum(kernel)) == 0:
        div = 1
    else:
        div = 1/abs(np.sum(kernel))

    for y in range(0, size[0]):
        for x in range(0, size[1]):
            mat = image[y:y+k, x:x+k]
            img[y, x] = np.sum(np.multiply(mat, kernel))*div

    return img

def sobel(image, threshold):

    deltax = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    deltay = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    
    Gaussian = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])

    image = convolve(image, Gaussian)

    newx = convolve(image, deltax)
    newy = convolve(image, deltay)

    gradMag = np.sqrt(np.square(newx) + np.square(newy))
    #gradMag *= 255.0 / gradMag.max()
    for y in range (0, gradMag.shape[0]):
        for x in range (0, gradMag.shape[1]):
            if gradMag[y, x] < threshold:
                gradMag[y, x] = 0
            else:
                gradMag[y, x] = 255

    #gradAngle = (np.arctan(np.divide(newy, newx, where=newx!=0)) * 180) / np.pi
    #gradAngle = cv2.phase(newx, newy, angleInDegrees = True)
    gradAngle = np.arctan2(newy, newx, where=newx!=0) * (180 / np.pi) %180
    # for y in range (0, gradAngle.shape[0]):
    #     for x in range (0, gradAngle.shape[1]):
    #         if gradAngle[y, x] > 128:
    #             gradAngle[y, x] = 255
    #         else:
    #             gradAngle[y, x] = 0

    cv2.imwrite("Horizontal Edge.jpg", newx)
    cv2.imwrite("Vertical Edge.jpg", newy)
    cv2.imwrite("Gradient Magnitude.jpg", gradMag)
    cv2.imwrite("Gradient Angle.jpg", gradAngle)

    return gradMag, gradAngle

def hough(col_image, gradMag, threshold, region, radius):
    height, width = gradMag.shape

    [R_max, R_min] = radius

    R = R_max - R_min

    houghSpace = np.zeros((R_max, height + 2*R_max, width + 2*R_max))
    h = np.zeros((R_max, height + 2*R_max, width + 2*R_max))

    theta = np.arange(0, 360) * np.pi / 180
    edges = np.argwhere(gradMag[:,:])

    for val in range(R):
        print(val+1, "/", R)
        r = R_min + val
        bprint = np.zeros((2*(r + 1), 2*(r + 1)))
        (h, w) = (r+1, r+1)
        for angle in theta:
            x  = int(np.round(r*np.cos(angle)))
            y  = int(np.round(r*np.sin(angle)))
            bprint[h+x, w+y] = 1
        const = np.argwhere(bprint).shape[0]
        for x, y in edges:
            X = [x-h+R_max,x+h+R_max]                                           #Computing the extreme X values
            Y= [y-w+R_max,y+w+R_max]                                           #Computing the extreme Y values
            houghSpace[r,X[0]:X[1],Y[0]:Y[1]] += bprint
        houghSpace[r][houghSpace[r]<threshold*const/r] = 0

    for r,x,y in np.argwhere(houghSpace):
        temp = houghSpace[r-region:r+region,x-region:x+region,y-region:y+region]
        try:
            p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
        except:
            continue
        h[r+(p-region),x+(a-region),y+(b-region)] = 1

    houghSpace = h[:,R_max:-R_max,R_max:-R_max]

    circleCoordinates = np.argwhere(houghSpace)                                          #Extracting the circle information
    for r,x,y in circleCoordinates:
        cv2.circle(col_image, (int(y), int(x)), int(r), (255, 0, 0), thickness = 3)
        cv2.circle(col_image, (int(y), int(x)), 1, (0, 0, 255), thickness = 2)
    
    return col_image

def displayCircles(colour, A):
    circleCoordinates = np.argwhere(A)                                          #Extracting the circle information
    circle = []
    for r,x,y in circleCoordinates:
        cv2.circle(colour, (int(y), int(x)), int(r), (0,255,0), thickness = 3)
        cv2.circle(colour, (int(y), int(x)), 1, (0, 255, 0), thickness = 2)
        
imagename = "coins1"
colour = cv2.imread(imagename+".png", cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)

threshold = 90
gradMag, gradAngle = sobel(image, threshold)

c1Config = (colour, gradMag, 33, 15, [50, 3])
c2Config = (colour, gradMag, 45, 15, [100, 3])
c3Config = (colour, gradMag, 30, 90, [60, 3])

finimg = hough(c1Config[0], c1Config[1], c1Config[2], c1Config[3], c1Config[4])
#finimg = hough2(c2Config[0], c2Config[1], c2Config[2], c2Config[3], c2Config[4])
#finimg = hough2(c3Config[0], c3Config[1], c3Config[2], c3Config[3], c3Config[4])

cv2.imwrite(imagename+".jpg", finimg)
cv2.imshow("Circle", finimg)
cv2.waitKey(0)
cv2.destroyAllWindows() 
