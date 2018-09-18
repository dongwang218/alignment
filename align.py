# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png

# import the necessary packages
import os, shutil
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
import numpy as np
from scipy import ndimage
from myutils import *
from flow import order_candidates

def revert_flow(gray, flow):
  h, w = flow.shape[:2]
  tmp = np.copy(flow)
  tmp[:,:,0] += np.arange(w)
  tmp[:,:,1] += np.arange(h)[:,np.newaxis]
  res = cv2.remap(gray, tmp, None, cv2.INTER_LINEAR)
  return res

def get_red_mask(img):
  bgr_inv = 255 - img
  hsv_inv = cv2.cvtColor(bgr_inv, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv_inv, np.array([90 - 10, 70, 50]), np.array([90 + 10, 255, 255]))
  return mask

def match_and_clean(first, second, binary, threshold, verbose, percentile):
  imageA = cv2.imread(first)
  max_width = imageA.shape[1]
  gray = enhance(imageA, binary)

  stitcher = Stitcher()

  diff = np.zeros(imageA.shape[:2]).astype(np.float)
  warped = [gray]
  for second_file in second:
    imageB = cv2.imread(second_file)
    imageB = imutils.resize(imageB, width=max_width)
    # stitch the images together to create a panorama
    transformed = stitcher.stitch([imageA, imageB], showMatches=False, reprojThresh=4.0)
    grayB = enhance(transformed, binary)

    flow = cv2.calcOpticalFlowFarneback(gray, grayB, None, 0.5, 2, 15, 3, 5, 1.1, 0)
    flowB = revert_flow(transformed, flow)
    warped_gray = enhance(flowB, binary)
    warped.append(warped_gray)
    diff += diff_image(gray, warped_gray)
  diff /= len(second)

  # print(np.histogram(diff, 100))

  others = np.median(np.array(warped), axis=0).astype(np.uint8)
  #others = ((np.max(np.array(warped), axis=0) + np.mean(np.array(warped), axis=0)) / 2).astype(np.uint8)
  others_percentile = np.percentile(np.array(warped), percentile, axis=0).astype(np.uint8)

  black, white = estimate_black_white(gray)
  print('black', black, 'white', white)

  # compute a threshold
  if threshold == 0:
    threshold = threshold_for_most_dark(diff)
  mask = diff > threshold
  red_mask = get_red_mask(imageA)
  total_mask = np.logical_or(mask, red_mask)
  black_mask = np.logical_and(total_mask, others < (int(black) + white) / 2)
  white_mask = np.logical_and(total_mask, others > (int(black) + white) / 2)

  print('total to remove black pixels', np.sum(black_mask), 'white pixels', np.sum(white_mask))
  corrected = np.copy(gray)
  corrected[black_mask] = others[black_mask]
  corrected[white_mask] = ndimage.median_filter(gray, size=20)[white_mask]
  #corrected[total_mask] = ndimage.median_filter(gray, size=20)[total_mask]

  #cv2.imshow('flow', draw_flow(grayB, flow))
  # show the images
  if verbose:
    cv2.imshow("Image A", imutils.resize(imageA, width=1280))
    cv2.imshow("corrected", imutils.resize(corrected, width=1280))
    cv2.imshow('diff', imutils.resize(diff.astype(np.uint8), width=1280))
    cv2.imshow('mask', imutils.resize(mask.astype(np.uint8)*255, width=1280))
    cv2.imshow('blackmask', imutils.resize(black_mask.astype(np.uint8)*255, width=1280))
    cv2.imshow('whitemask', imutils.resize(white_mask.astype(np.uint8)*255, width=1280))
    cv2.imshow("others", imutils.resize(others, width=1280))
    cv2.waitKey(0)

  return corrected, others_percentile, diff

if __name__ == '__main__':

  # construct the argument parse and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('--max_width', type=int, default=800, help='width to resize smaller is faster')
  ap.add_argument('--block_size', type=int, default=20, help='group images to speed up')
  ap.add_argument('-n', '--number', type=int, default=10, help='num of others to consider')
  ap.add_argument('-t', '--threshold', type=int, default=0, help='threshold to detect things to remove, the bigger the more to be removed. 0 means auto detect.')

  ap.add_argument('-v', '--verbose', type=int, default=0, help='verbose')
  ap.add_argument('-b', '--binary', type=int, default=0, help='work on binary')
  ap.add_argument('--out_dir', required=True, help='save')
  ap.add_argument('--max_images', type=int, default=60, help='only consider up to 60 images')
  ap.add_argument('--working_dir', type=str, default='/tmp/mysplit', help='working dir')
  ap.add_argument('--try_split', type=int, default=1, help='whether try to split in half')
  ap.add_argument('--percentile', type=float, default=80, help='percentile to pick as output')

  ap.add_argument("files", nargs='*',
    help="path to the second images")
  args = ap.parse_args()

  dir = args.working_dir
  if os.path.exists(dir):
    shutil.rmtree(dir)
  os.makedirs(dir)
  new_files = []
  for f in args.files:
    img = cv2.imread(f)
    if img is None:
      print('missing %s' % f)
      continue
    img = remove_black_border(img)
    nf = os.path.join(dir, os.path.basename(f))
    new_files.append(nf)
    cv2.imwrite(nf, img)
    #h,w = img.shape[:2]
    #part = img[:, :w//2] if args.half == 'left' else img[:, w//2:]
  args.files = new_files
  print('new files', args.files)

  candidates = order_candidates(args.files[:args.max_images], args.max_width, args.block_size)
  print(candidates)
  best_index = candidates[0]
  first = args.files[best_index]
  second = []
  for i in candidates[1:]:
    second.append(args.files[i])

  print('first', first)
  print('second', ' '.join(second))

  # load the two images and resize them to have a width of 400 pixels
  # (for faster processing)
  imageA = cv2.imread(first)
  max_width = imageA.shape[1]
  gray = enhance(imageA, args.binary)
  split_index = is_gray_double_page(gray, args.binary)

  if args.try_split != 1 or split_index is None:
    corrected, others, diff = match_and_clean(first, second[:args.number], args.binary, args.threshold, args.verbose, args.percentile)
  else:
    dir_left = os.path.join(args.working_dir, 'left')
    dir_right = os.path.join(args.working_dir, 'right')
    os.makedirs(dir_left)
    os.makedirs(dir_right)
    left = []
    right = []
    left_file = os.path.join(dir_left, os.path.basename(first))
    right_file = os.path.join(dir_right, os.path.basename(first))
    left.append(left_file)
    right.append(right_file)
    cv2.imwrite(left_file, imageA[:, :split_index])
    cv2.imwrite(right_file, imageA[:, split_index:])

    stitcher = Stitcher()

    for second_file in second:
      imageB = cv2.imread(second_file)
      imageB = imutils.resize(imageB, width=max_width)
      # stitch the images together to create a panorama
      transformed = stitcher.stitch([imageA, imageB], showMatches=False, reprojThresh=4.0)
      n_split_index = is_gray_double_page(transformed)
      if split_index is None:
        n_split_index = split_index
      left_file = os.path.join(dir_left, os.path.basename(second_file))
      right_file = os.path.join(dir_right, os.path.basename(second_file))
      left.append(left_file)
      right.append(right_file)
      cv2.imwrite(left_file, transformed[:, :n_split_index])
      cv2.imwrite(right_file, transformed[:, n_split_index:])
      if len(left) > args.number:
        break
    corrected_left, others_left, diff_left = match_and_clean(left[0], left[1:], args.binary, args.threshold, args.verbose, args.percentile)
    corrected_right, others_right, diff_right = match_and_clean(right[0], right[1:], args.binary, args.threshold, args.verbose, args.percentile)
    corrected = np.concatenate((corrected_left, corrected_right), axis=1)
    others = np.concatenate((others_left, others_right), axis=1)
    diff = np.concatenate((diff_left, diff_right), axis=1)

  np.save('diff.npy', diff)
  cv2.imwrite('corrected.png', corrected)
  if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
  outfile = os.path.join(args.out_dir, os.path.basename(first).split('.')[0]+'.png')
  cv2.imwrite(outfile, corrected)
  cv2.imwrite(outfile.split('.')[0]+'-binary.png', enhance(corrected))
  cv2.imwrite(outfile.split('.')[0]+'-median.png', others)
  cv2.imwrite(outfile.split('.')[0]+'-median-binary.png', enhance(others))
