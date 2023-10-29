import cv2
import numpy as np

left_top_x = 0
left_bottom_x = 0
right_top_x = 450
right_bottom_x = 450

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

while True:
    ret, frame = cam.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    width = 300 * 2
    height = 180 * 2
    frame = cv2.resize(frame, (width, height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    arr = np.zeros((height, width), dtype=np.uint8)

    upper_left = (width * 0.45, height * 0.76)
    upper_right = (width * 0.55, height * 0.76)
    lower_left = (0, height)
    lower_right = (width, height)

    trapezoid_array = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)
    window_bounds = np.array([(width, 0), (0, 0), (0, height), (width, height)], dtype=np.float32)
    trapezoid_array_toFloat = np.float32(trapezoid_array)
    stretching_matrix = cv2.getPerspectiveTransform(trapezoid_array_toFloat, window_bounds)

    arr = cv2.fillConvexPoly(arr, trapezoid_array, 1)
    just_road = arr * gray
    top_down_view = cv2.warpPerspective(just_road, stretching_matrix, (width, height))
    blurred = cv2.blur(top_down_view, ksize=(5, 5))

    sobel_vertical_matrix = np.float32([[-1, -2, -1],
                                        [0, 0, 0],
                                        [1, 2, 1]])
    blurred_32f = np.float32(blurred)
    sobel_horizontal_matrix = np.transpose(sobel_vertical_matrix)
    sobel_vertical = cv2.filter2D(blurred_32f, -1, sobel_vertical_matrix)
    sobel_horizontal = cv2.filter2D(blurred_32f, -1, sobel_horizontal_matrix)
    final_sobel = np.sqrt(sobel_horizontal ** 2 + sobel_vertical ** 2)
    edge_detection = cv2.convertScaleAbs(final_sobel)

    binarized_edge = cv2.convertScaleAbs((edge_detection > 55) * 255)

    binarized_edge_copy = binarized_edge.copy()
    binarized_edge_copy[:, :int(width * 0.10)] = 0
    binarized_edge_copy[:, -int(width * 0.10):] = 0

    split_col = width // 2
    left_half = binarized_edge_copy[:, :split_col]
    right_half = binarized_edge_copy[:, split_col:]
    argwhere_left = np.argwhere(left_half > 1)
    argwhere_right = np.argwhere(right_half > 1)

    left_xs = argwhere_left[:, 1]
    left_ys = argwhere_left[:, 0]
    right_xs = argwhere_right[:, 1] + width // 2
    right_ys = argwhere_right[:, 0]

    left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    left_top_y = 0
    left_bottom_y = height
    right_top_y = 0
    right_bottom_y = height

    left_top_xx = int((left_top_y - left_line[0]) / left_line[1])
    left_bottom_xx = int((left_bottom_y - left_line[0]) / left_line[1])
    right_top_xx = int((right_top_y - right_line[0]) / right_line[1])
    right_bottom_xx = int((right_bottom_y - right_line[0]) / right_line[1])

    if 0 <= left_top_xx < width // 2:
        left_top_x = left_top_xx

    if 0 <= left_bottom_xx < width // 2:
        left_bottom_x = left_bottom_xx

    if width // 2 <= right_bottom_xx < width:
        right_bottom_x = right_bottom_xx

    if width // 2 < right_top_xx < width:
        right_top_x = right_top_xx

    print(right_top_xx, right_bottom_xx)
    cv2.line(binarized_edge, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), (200, 0, 0), 5)
    cv2.line(binarized_edge, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), (100, 0, 0), 5)

    final_left_lane = np.zeros((width, height), dtype=np.uint8)

    cv2.line(final_left_lane, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), (255, 0, 0), 5)

    stretching_matrix_reversed = cv2.getPerspectiveTransform(window_bounds, trapezoid_array_toFloat)
    final_left_lane = cv2.warpPerspective(final_left_lane, stretching_matrix_reversed, (width, height))

    final_right_lane = np.zeros((width, height + 150), dtype=np.uint8)
    cv2.line(final_right_lane, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), (100, 0, 0), 10)
    cv2.imshow("finalright", final_right_lane)

    final_right_lane = cv2.warpPerspective(final_right_lane, stretching_matrix_reversed, (width, height))

    argwhere_left = np.argwhere(final_left_lane > 1)
    argwhere_right = np.argwhere(final_right_lane > 1)
    # print(right_ys,right_xs)

    right_half = final_right_lane[:, split_col:]

    left_xs = argwhere_left[:, 1]
    left_ys = argwhere_left[:, 0]
    right_xs = argwhere_right[:, 1]
    right_ys = argwhere_right[:, 0]

    frame_final = frame.copy()
    frame_final[left_ys, left_xs] = (50, 50, 250)
    frame_final[right_ys, right_xs] = (50, 250, 50)

    cv2.imshow('Original', frame)
    cv2.imshow('grayscale', gray)
    cv2.imshow('sosea', just_road)
    cv2.imshow("intors", top_down_view)
    cv2.imshow("blurat", blurred)
    cv2.imshow("binarized edge", binarized_edge)
    cv2.imshow("Semi gata", final_left_lane)
    cv2.imshow("Semi gataa", final_right_lane)
    cv2.imshow("final", frame_final)

cam.release()
cv2.destroyAllWindows()
