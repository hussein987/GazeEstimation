import numpy as np
import dlib
import cv2
import torch
from CNN import CNN1
import torchvision.transforms as T
from PIL import Image


transforms = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((5, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 5):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def f(v, c):
    if v>=c:
        return v+100
    else:
        return v


def draw_arrow(img, points, translation):
    # img = cv2.imread('left.jpg')
    c_x, c_y = points[0]
    x, y = points[1][0], points[1][1]
    # print(x, y)
    translation_x, translation_y = translation
    cv2.arrowedLine(img, (c_x, c_y), (c_x+x*80+translation_x, c_y+y*80+translation_y), (0, 0, 255), 2)
    # cv2.circle(img, (c_x+x*160+translation_x, c_y+y*160+translation_y), 2, (255, 0, 255), 2)

    # cv2.imshow('name', img)
    # cv2.waitKey(0)


def return_eyes_region(image, shape, shift_y, shift_x):
    left_eye = image[shape[0][1] - shift_y:shape[0][1] + shift_y, shape[1][0] - shift_x:shape[0][0] + shift_x]
    # cv2.imwrite('left.jpg', left_eye)
    # cv2.imshow('name', left_eye)
    # cv2.waitKey(0)
    right_eye = image[shape[2][1] - shift_y:shape[2][1] + shift_y, shape[2][0] - shift_x:shape[3][0] + shift_x]
    # left_eye = Image.open('real2.jpg').convert("RGB")
    return left_eye, right_eye


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../models/shape_predictor_5_face_landmarks.dat')

cap = cv2.VideoCapture(0)

model = CNN1()
model.load_state_dict(torch.load('../models/model2.pt', map_location=torch.device('cpu')))
model.eval()

j = -1
while cap.isOpened():
    j += 1

    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    image.flags.writeable = False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    left_eye = None
    right_eye = None

    # loop over the face detections
    for (i, rect) in enumerate(rects):

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = rect_to_bb(rect)

        # and draw them on the image
        image.setflags(write=1)
        shape = shape[:4]
        shift = 20

        # left_eye = image[shape[0][1]-shift:shape[0][1]+shift, shape[1][0]:shape[0][0]]
        # right_eye = image[shape[2][1]-shift:shape[2][1]+shift, shape[2][0]:shape[3][0]]
        left_eye, right_eye = return_eyes_region(image, shape, 30, 20)

        left_eye_center = (int(shape[1][0]+(shape[0][0]-shape[1][0])/2), shape[0][1])
        right_eye_center = (int(shape[2][0]+(shape[3][0]-shape[2][0])/2), shape[2][1])

        # cv2.circle(image, left_eye_center, 1, (0, 255, 0), -1)
        # cv2.circle(image, right_eye_center, 1, (0, 255, 0), -1)
        #
        # for (x, y) in shape[:2]:
        #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        # for (x, y) in shape[2:]:
        #     cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

        # Predict once per 2 frames
        if j % 1 == 0:

            # Left Eye
            PIL_image = Image.fromarray(np.uint8(left_eye*128)).convert('RGB')
            # PIL_image = Image.fromarray((left_eye * 128).astype(np.uint8)).convert('RGB')
            # PIL_image = Image.fromarray(left_eye*128).convert('RGB')
            img = transforms(PIL_image)
            # img = transforms(left_eye)

            # test_img = cv2.cvtColor(np.float32(img.permute(1, 2, 0)))
            # cv2.imshow('test', np.float32(img.permute(1, 2, 0)))
            # cv2.imwrite('test.png', cv2.convertScaleAbs(np.float32(img.permute(1, 2, 0)), alpha=(255.0)))
            # cv2.waitKey(0)

            out = model(img.unsqueeze(0))
            translation_x = shape[1][0] - 20
            translation_y = shape[1][1] - 30
            # test_img = T.ToPILImage()(img).convert("RGB")
            # test = np.array(test_img)
            # open_cv_image = test_img[:, :, ::-1].copy()

            draw_arrow(left_eye, [(left_eye_center[0]-translation_x, left_eye_center[1]-translation_y), (out['label1'], out['label2'])], (0, 0))

            # break
            # print((out['label1']*400, out['label2']*400))
            # print(left_eye.shape)
            # cv2.circle(test_img, (64+out['label1']*400, 64+out['label2']*400), 1, (0, 255, 0), 10)
            # cv2.imshow('test', open_cv_image)

            # test_img = cv2.cvtColor(np.float3 2(img.permute(1, 2, 0)), cv2.COLOR_RGB2GRAY)
            # cv2.imshow('test', test_img)

            # Right Eye
            PIL_imageR = Image.fromarray(np.uint8(right_eye*128)).convert('RGB')
            imgR = transforms(PIL_imageR)
            outR = model(imgR.unsqueeze(0))
            translation_x = shape[2][0] - 20
            translation_y = shape[2][1] - 30
            draw_arrow(right_eye, [(right_eye_center[0]-translation_x, right_eye_center[1]-translation_y), (outR['label1'], outR['label2'])], (0, 0))

    cv2.imshow("Output", cv2.flip(image, 1))
    # cv2.imshow("Output", image)

    try:
        window_x = 900
        window_y = 170

        left_eye_window = 'Left eye'
        cv2.moveWindow(left_eye_window, window_x, window_y)
        cv2.imshow(left_eye_window, cv2.flip(left_eye, 1))
        # cv2.imshow(left_eye_window, left_eye)

        right_eye_window = 'Right eye'
        cv2.moveWindow(right_eye_window, window_x+70, window_y)
        cv2.imshow(right_eye_window, cv2.flip(right_eye, 1))
        # cv2.imshow(right_eye_window, right_eye)
    except:
        print('can\'t see eyes')

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
