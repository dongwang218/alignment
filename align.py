# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
import numpy as np
from myutils import *

def get_red_mask(img):
  bgr_inv = 255 - img
  hsv_inv = cv2.cvtColor(bgr_inv, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv_inv, np.array([90 - 10, 70, 50]), np.array([90 + 10, 255, 255]))
  return mask

if __name__ == '__main__':

  # construct the argument parse and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-f", "--flow", required=True,
    help="path to the smallest_flow.npy")
  ap.add_argument('-n', '--number', type=int, default=10, help='num of others to consider')
  ap.add_argument("files", nargs='*',
    help="path to the second images")
  args = ap.parse_args()

  total_flow = np.load(args.flow)
  total_sum = np.sum(total_flow, axis=-1)
  print(total_sum)
  best_index = np.argmin(total_sum)
  print(best_index)
  best_indexes = np.argsort(total_flow[best_index])

  assert(best_indexes[0] == best_index)

  first = args.files[best_index]
  second = []
  for i in best_indexes[1:(args.number+1)]:
    second.append(args.files[i])


  print('first', first)
  print('second', ' '.join(second))

  # load the two images and resize them to have a width of 400 pixels
  # (for faster processing)
  imageA = cv2.imread(first)
  max_width = imageA.shape[1]
  gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)

  stitcher = Stitcher()

  diff = np.zeros(imageA.shape[:2]).astype(np.float)
  warped = []
  for second_file in second:
    imageB = cv2.imread(second_file)
    imageB = imutils.resize(imageB, width=max_width)
    # stitch the images together to create a panorama
    transformed = stitcher.stitch([imageA, imageB], showMatches=False, reprojThresh=4.0)
    grayB = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(grayB, gray, None, 0.5, 5, 15, 3, 5, 1.2, 0)
    flowB = warp_flow(transformed, flow)
    warped_gray = cv2.cvtColor(flowB, cv2.COLOR_BGR2GRAY)
    warped.append(warped_gray)
    diff += diff_image(gray, warped_gray)
  diff /= len(second)

  print(np.histogram(diff, 100))

  others = (np.max(np.array(warped), axis=0) + np.mean(np.array(warped), axis=0)) / 2

  corrected = np.copy(gray)
  mask = diff > 40
  red_mask = get_red_mask(imageA)
  total_mask = np.logical_or(mask, red_mask)
  corrected[total_mask] = others[total_mask]

  # show the images
  cv2.imshow("Image A", imageA)
  cv2.imshow("corrected", corrected)
  cv2.imshow('diff', diff.astype(np.uint8))

  np.save('diff.npy', diff)
  cv2.imwrite('corrected.png', corrected)
  cv2.waitKey(0)
