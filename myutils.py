import numpy as np
import cv2
from imutils import perspective
import imutils

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def mix_image(imgA, imgB, alpha=0.5):
  result = np.copy(imgA)
  result = (result * alpha + (1-alpha) * imgB).astype(np.uint8)
  return result

def diff_image(imgA, imgB, is_gray=True):
  if is_gray:
    grayA = imgA
    grayB = imgB
  else:
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
  return np.abs(grayA.astype(np.int) - grayB).astype(np.uint8)

def remove_black_border(img, black=80):
  # use binary threshold, all pixel that are beyond 3 are made white
  _, thresh_original = cv2.threshold(img, black, 255, cv2.THRESH_BINARY)
  if thresh_original is None:
    import pdb
    pdb.set_trace()
  thresh = np.copy(thresh_original[:,:,0])

  # Now find contours in it.
  _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  if contours:
    # get cnt with largest height
    index = None
    height = None
    for i, cnt in enumerate(contours):
      ctr = cv2.boundingRect(cnt)
      if height is None or height < ctr[3]:
        index = i
        height = ctr[3]
    cnt = contours[index]
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    if len(approx) == 4:
      warped = perspective.four_point_transform(img, approx.reshape(4, 2))
      return warped

  return np.copy(img)

def pyimage_remove_black_border(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (5, 5), 0)
  edged = cv2.Canny(gray, 75, 200)

  cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
  cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

  # loop over the contours
  for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
      warped = perspective.four_point_transform(image, approx.reshape(4, 2))
      return warped

  return np.copy(image)

def enhance(img, binary=True):
  if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  else:
    gray = np.copy(img)
  if binary:
    black, white = estimate_black_white(gray)
    print('black', black, 'white', white)
    threshold = (int(black)+white)/2
    gray[gray < threshold] = 0
    gray[gray > threshold] = 255
  return gray


def is_gray_double_page(gray, is_binary=False):
  if not is_binary:
    binary = enhance(gray, True)
  else:
    binary = gray
  h, w = binary.shape
  total = np.sum(binary[(h//8):(-h//8), (w//2-30):(w//2+30)] == 0, axis = 0)
  min = np.min(total)
  ok = min < 10
  if ok:
    index = int(np.mean(np.where(total == min)))
    return (w//2-30) + index
  else:
    return None

def estimate_black_white(gray, distance=50):
  diff = 0
  h, w = gray.shape[:2]
  cx = w // 2
  cy = h // 2
  cw = w // 8
  ch = h // 8
  while diff < distance and cw < w and ch < h:
    sorted = np.sort(gray[cy-ch:cy+ch, cx-cw:cx+cw], axis=None)
    white = sorted[-len(sorted)//20]
    black = sorted[len(sorted)//100]
    diff = white-black
    cw += w // 8
    ch += h // 8

  return black, white

def threshold_for_most_dark(diff, verbose=False):

  background = np.median(diff)
  f_diff = diff.flatten()
  f_diff = f_diff[f_diff > background]
  f_diff = np.sort(f_diff)

  if verbose:
    import matplotlib.pyplot as plt
    n, bins, patches = plt.hist(f_diff)
    plt.show()

  bound = max(1, f_diff.shape[0] // 100)
  diff_b = f_diff[bound]
  diff_w = f_diff[-bound]
  threshold = (diff_b+diff_w)//2
  print('diff black', diff_b, 'diff white', diff_w, 'threshold', threshold)
  return threshold
