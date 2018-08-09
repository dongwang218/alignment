'''
compute the best image and the best other candidates
'''
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
      flowB = cv2.calcOpticalFlowFarneback(grays[i], grays[j], None, 0.5, 2, 15, 3, 5, 1.1, 0)#0.5, 5, 15, 3, 5, 1.2, 0)
      total = np.mean(np.linalg.norm(flowB, axis=-1))
      total_flow[i, j] = total

  total_sum = np.sum(total_flow, axis=-1)
  print(total_sum)
  best_index = np.argmin(total_sum)
  return best_index, total_sum[best_index]

def smallest_candidates(images, best_index):
  '''It is unclear whether we should do best -> others, or other -> best; or even should we use the smallest or the largest'''
  stitcher = Stitcher()

  imageA = images[best_index]
  gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
  grays = []
  for index, imageB in enumerate(images):
    if index == best_index: continue

    transformed = stitcher.stitch([imageA, imageB], showMatches=False, reprojThresh=4.0)
    grayB = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    grays.append(grayB)

  total_flow = np.zeros((len(grays)), dtype = np.float)
  for i in range(len(grays)):
    flowB = cv2.calcOpticalFlowFarneback(grays[i], gray, None, 0.5, 5, 15, 3, 5, 1.2, 0)
    total = np.mean(np.linalg.norm(flowB, axis=-1))
    total_flow[i] = total

  best_indexes = np.argsort(total_flow)
  best_indexes[best_indexes >= best_index] += 1

  return np.hstack(([best_index], best_indexes))

def order_candidates(files, max_width, block_size):
  images = []
  for file in files:
    imageA = cv2.imread(file)
    imageA = imutils.resize(imageA, width=max_width)
    images.append(imageA)

  block_size = min(block_size, len(images))
  slice = range(0, len(images), block_size)
  if (len(images)-slice[-1]) > block_size // 2:
    slice.append(len(images))
  else:
    slice[-1] = len(images)
  print('slice', slice)

  best_index = None
  best_flow = None
  for i in range(0, len(slice)-1):
    index, flow = smallest_flow(images[slice[i]:slice[i+1]])
    if best_flow is None or best_flow > flow:
      best_index = index + slice[i]
      best_flow = flow

  print('best_index', best_index)
  # find the best candidates
  ordered = smallest_candidates(images, best_index)
  return ordered

if __name__ == '__main__':

  # construct the argument parse and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('--max_width', type=int, default=800, help='width to resize smaller is faster')
  ap.add_argument('--block_size', type=int, default=10, help='group images to speed up')
  ap.add_argument("second", nargs='*',
    help="path to the second images")
  args = ap.parse_args()

  ordered = order_candidates(args.second, args.max_width, args.block_size)

  np.save('smallest_flow.npy', ordered)
  print(ordered)
  for i in ordered:
    print(args.second[i])
