# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png

# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
import numpy as np

def smallest_flow(images):
  stitcher = Stitcher()

  imageA = images[0]
  gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
  grays = [gray]
  for imageB in images[1:]:
    # stitch the images together to create a panorama
    transformed = stitcher.stitch([imageA, imageB], showMatches=False, reprojThresh=4.0)
    grayB = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    grays.append(grayB)

  total_flow = np.zeros((len(grays), len(grays)), dtype = np.float)
  for i in range(len(grays)):
    for j in range(len(grays)):
      flowB = cv2.calcOpticalFlowFarneback(grays[i], grays[j], None, 0.5, 5, 15, 3, 5, 1.2, 0)
      total = np.mean(np.linalg.norm(flowB, axis=-1))
      total_flow[i, j] = total

  total_sum = np.sum(total_flow, axis=-1)
  print(total_sum)
  best_index = np.argmin(total_sum)
  print(best_index)
  best_indexes = np.argsort(total_flow[best_index])

  print('best is %s' % (total_sum[best_index]))

  return total_flow, best_index, best_indexes

if __name__ == '__main__':

  # construct the argument parse and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("second", nargs='*',
    help="path to the second images")
  args = ap.parse_args()

  images = []
  for file in args.second:
    imageA = cv2.imread(file)
    imageA = imutils.resize(imageA, width=1280)
    images.append(imageA)

  total_flow, best_index, best_indexes = smallest_flow(images)
  np.save('smallest_flow.npy', total_flow)

  print(args.second[best_index])
  ordered = []
  for i in best_indexes:
    ordered.append(args.second[i])
    print(args.second[i])
