from threading import Thread
import cv2, time
import numpy as np
from opts import get_opts
from planarH import computeH_ransac, compositeH

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.opts = get_opts()
        self.board = np.zeros((self.opts.grid_size, self.opts.grid_size))
        self.pixelatedFrame = np.zeros((self.opts.grid_size, self.opts.grid_size, 3))
        self.visual_board = np.zeros((self.opts.grid_size, self.opts.grid_size, 3))
        self.H2to1 = np.load('H2to1.npy')
        self.mask = np.load('mask.npy')
        self.unwarped = cv2.resize(cv2.imread('../data/goboard8x8.jpg', -1), (512, 512))
        self.board_size = (2 * self.opts.grid_size) + 1

        positions = np.arange(1, self.board_size, 2)
        j, i = np.array(np.meshgrid(positions, positions))
        coordinates = np.dstack((i, j)).reshape(self.opts.grid_size ** 2, 2)
        self.starts = np.array((self.opts.gui_size // self.board_size) * coordinates)
        self.ends = np.array(((self.opts.gui_size // self.board_size) * (coordinates + 1)) + 1)

        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.opts.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.opts.frame_height)
        if self.capture.isOpened():
            (self.status, self.prev_frame) = self.capture.read()
            self.last_valid_frame = self.prev_frame
            gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            self.prev_frame = gray
        self.static = False
        self.board_updated = True
        self.static_recorded = False
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                frameDelta = cv2.absdiff(self.prev_frame, gray)
                self.thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
                self.thresh = cv2.dilate(self.thresh, None, iterations=2)
                self.movement = np.sum(self.thresh)
                # self.movement = np.sum(cv2.absdiff(gray, self.prev_frame))
                print(self.movement)
                self.prev_frame = gray
                if self.movement <= 0 and self.static is False:
                    self.static = True
                    t = time.time()
                elif self.movement > 0 and self.static is True:
                    self.static = False
                    self.static_recorded = False
                elif self.movement <= 0 and self.static is True:
                    elapsedTime = time.time() - t
                    if elapsedTime > 3 and not self.static_recorded:
                        self.last_valid_frame = self.frame
                        self.static_recorded = True
                        self.board_updated = False
                        print("updated frame, elapsed time = ", elapsedTime)
            time.sleep(.05)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        cv2.imshow('movement', self.thresh)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def get_last_valid_frame(self):
        return self.last_valid_frame

    def get_board(self):
        return self.board

    def show_board(self):
        cv2.imshow('pixelated', self.pixelatedFrame)
        cv2.imshow('visual_board', self.visual_board)

    def update_board(self):
        if not self.board_updated:
            modFrame = compositeH(self.H2to1, self.last_valid_frame, self.unwarped) * \
                       self.mask.reshape((self.mask.shape[0], self.mask.shape[1], 1)).astype('uint8')

            averages = np.zeros((self.opts.grid_size ** 2, 3))

            for k in range(modFrame.shape[2]):
                channel = modFrame[:, :, k]
                for i in range(self.starts.shape[0]):
                    frame_section = np.array(channel[self.starts[i, 0]:self.ends[i, 0], self.starts[i, 1]:self.ends[i, 1]]).flatten()
                    mask_section = np.array(self.mask[self.starts[i, 0]:self.ends[i, 0], self.starts[i, 1]:self.ends[i, 1]]).flatten()
                    averages[i, k] = np.sum(frame_section) / np.sum(mask_section)

            averages = averages.reshape(self.opts.grid_size, self.opts.grid_size, 3)

            red_dist = np.sum(np.square(averages - self.opts.red), axis=2)
            black_dist = np.sum(np.square(averages - self.opts.black), axis=2)
            white_dist = np.sum(np.square(averages - self.opts.white), axis=2)

            board = np.zeros((self.opts.grid_size, self.opts.grid_size))
            visual_board = np.zeros(averages.shape)

            for i in range(averages.shape[0]):
                for j in range(averages.shape[1]):
                    if white_dist[i, j] < red_dist[i, j] and white_dist[i, j] < black_dist[i, j]:
                        board[i, j] = 2
                        visual_board[i, j, :] = [255, 255, 255]
                    elif black_dist[i, j] < red_dist[i, j] and black_dist[i, j] < white_dist[i, j]:
                        board[i, j] = 1
                        visual_board[i, j, :] = [0, 0, 0]
                    else:
                        board[i, j] = 0
                        visual_board[i, j, :] = [0, 0, 255]

            self.board = board
            averages = np.array(averages).astype('uint8')
            self.pixelatedFrame = np.array(cv2.resize(averages, (self.opts.gui_size, self.opts.gui_size), interpolation=cv2.INTER_NEAREST)).astype('uint8')
            self.visual_board = np.array(cv2.resize(visual_board, (self.opts.gui_size, self.opts.gui_size), interpolation=cv2.INTER_NEAREST)).astype('uint8')
            self.board_updated = True
