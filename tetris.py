from hashlib import new
import cv2
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import style
from copy import deepcopy

BLKSIZE = 20
HEIGHT = 20
WITDH = 10

'''
   Simple game engine
'''
class Tetris:
   '''
      We need one-sided tetrominos as we can't transform free tetronimos L -> J or S -> Z from rotation. 
      This is due to chirality phenomenon.
   '''
   tetrominos = [
      # O shape
      [[1, 1],
       [1, 1]],
      # T shape
      [[2, 2, 2],
       [0, 2, 0]],
      # L shape
      [[3, 0],
       [3, 0],
       [3, 3]],
      # J shape
      [[0, 4],
       [0, 4],
       [4, 4]],
      # S shape
      [[0, 5, 5],
       [5, 5, 0]],
      # Z shape
      [[6, 6, 0],
       [0, 6, 6]],
      # I shape
      [[7],
       [7],
       [7],
       [7]],
   ]
   ''' 
      Follows the same order as the tetronimos array
   '''
   tetromino_color = [
      (0, 0, 0),
      (255,255,0),
      (255,0,255),
      (255,215,0),
      (0,0,255),
      (0,255,0),
      (255,0,0),
      (135,206,250)
   ]

   '''
      Initatilze board. Side panel and text color go to the right of the board
   '''
   def __init__(self) -> None:
      self.text_color = (240,248,255)
      self.side_panel = np.ones((HEIGHT * BLKSIZE, WITDH * int(BLKSIZE / 2), 3), dtype=np.uint8) * np.array([95, 158, 160], dtype=np.uint8)
      self.reset()

   '''
      Resets the game when gameover = True. In training, it will return the state properties
      after losing.
   '''
   def reset(self):
      self.board = np.zeros((HEIGHT, WITDH), dtype=np.uint8)
      self.score = 0
      self.count = 0
      self.lines = 0
      self.indexes = list(range(len(self.tetrominos)))
      random.shuffle(self.indexes)
      self.idx = self.indexes.pop()
      self.tetromino = deepcopy(self.tetrominos[self.idx])
      self.current_pos = {'x': int(WITDH / 2) - int(len(self.tetrominos[0]) / 2), 'y': 0}
      self.gameover = False
      return self.state_properties(self.board)

   '''
      These heuristics of the game will be what our training model uses to asses performance.
         Complete Lines: We want to maximize this because it is the goal of the AI + more space
         Aggregate Height: We want to minimize this value because then we can drop more pieces
         Holes: We want to minimize this because the less holes the more lines we can complete
         Bumpiness: We want to mnimize this value so our board doesn't fill up in unwanted places

         Returns: tf float array with the 4 properties
   '''
   def state_properties(self, board):
      line_count, board = self.completed_lines(board)
      agg_height = sum(self.heights(board))
      holes = self.hole_count(board)
      bumpy_score = self.bumpiness(board)
      return np.array([line_count, holes, bumpy_score, agg_height])
      # return tf.constant([line_count, holes, bumpy_score, agg_height])
      
   '''
      Checking if ndarray in each row contains a 0. If so we want to remove it from our board.
      
      Returns the length of amount of lines deleted and the updated board without those lines.
   '''
   def completed_lines(self, board):
      completed_lines = []
      for i, row in enumerate(board[::-1]):
         if np.all(row):
            completed_lines.append(len(board) - i - 1)
      if len(completed_lines) > 0:
         board = self.remove_line(board, completed_lines)
      return len(completed_lines), board

   '''
      Flips the board on its side using zip, going through each column once a 1 appears at the top
      start counting holes if a 0 appears. 

      Returns all the holes found.
   '''
   def hole_count(self, board):
      holes = 0
      for col in np.stack(board, axis=1):
         valid = False
         for i in range(HEIGHT):
            if col[i] == 1:
               valid = True
            if valid == True and col[i] == 0:
               holes += 1
      return holes

   '''
      Queries where the board has values, it takes the max of each column and subtracts 20 to make the number it's true height.
      Used in getting aggregated heights and board bumpiness.

      Returns the max height of each column.
   '''
   def heights(self, board):
      return HEIGHT - np.where(board.any(axis=0), np.argmax(board, axis=0), HEIGHT) 
      
   '''
      Given all the heights, we sum up the absolute differences between all two adjacent columns

      Returns the 'bumpiness' of board.
   '''
   def bumpiness(self, board):
      col_heights = self.heights(board)
      lhs = col_heights[:-1]
      rhs = col_heights[1:]
      differences = np.abs(lhs - rhs)
      return np.sum(differences)

   '''
      Taking in a board and a list of row indincies, delete all the lines from the board and 
      with vstack add a new row of 0's to the top

      Returns the updated board with removed lines
   '''
   def remove_line(self, board, indices):
      for i in indices[::-1]:
         board = np.delete(board, i, 0)
         new_row = np.zeros((10), dtype=np.uint8)
         board = np.vstack([new_row, board])
      return board

   '''
      Takes a tetromino and rotates the array 90 degrees. This is done using rot90() from numpy.

      Returns rotated tetromino.
   '''
   def rotate(self, tetromino):
      return np.rot90(tetromino).tolist()

   '''
      Certain pieces can only really rotate a set amount of times
   '''
   def avaliable_rotations(self):
      if self.idx in [4,5,6]:
         return 2
      elif self.idx in [1,2,3]:
         return 4
      else:
         return 1

   def next_states(self):
      states = {}
      current_tetromino = deepcopy(self.tetromino)
      rotations = self.avaliable_rotations()
      for i in range(rotations):
         valid_positions = WITDH - len(current_tetromino[0]) + 1
         for x in range(valid_positions):
            tetromino = deepcopy(current_tetromino)
            pos= {'x': x, 'y': 0}
            while not self.check_collision(tetromino, pos):
               pos['y'] += 1
            self.truncate(tetromino, pos)
            board = self.store(tetromino, pos)
            states[(x, i)] = self.state_properties(board)
         current_tetromino = self.rotate(current_tetromino)
      return [(x[0],x[1]) for x in states.items()] # {k:p} -> [(k,p)]

   def current_board_state(self):
      board = deepcopy(self.board)
      for y in range(len(self.tetromino)):
         for x in range(len(self.tetromino[y])):
            board[y + self.current_pos['y']][x+ self.current_pos['x']] = self.tetromino[y][x]
      return board

   def new_piece(self):
      if not len(self.indexes):
         self.indexes = list(range(len(self.tetrominos)))
         random.shuffle(self.indexes)
      self.idx = self.indexes.pop()
      self.tetromino = deepcopy(self.tetrominos[self.idx])
      self.current_pos = {'x': int(WITDH / 2) - int(len(self.tetrominos[0]) / 2), 'y': 0}
      if self.check_collision(self.tetromino, self.current_pos):
         self.gameover = True

   def check_collision(self, tetromino, pos):
      new_y = pos['y'] + 1
      # try to convert using itertools
      for y in range(len(tetromino)):
         for x in range(len(tetromino[y])):
            if new_y + y > HEIGHT - 1 or self.board[new_y + y][pos['x'] + x] and tetromino[y][x]:
               return True
      return False

   def truncate(self, tetromino, pos):
      gameover = False
      last_collision_row = -1
      for y in range(len(tetromino)):
         for x in range(len(tetromino[y])):
            if self.board[pos["y"] + y][pos["x"] + x] and tetromino[y][x]:
               if y > last_collision_row:
                  last_collision_row = y
      if pos["y"] - (len(tetromino) - last_collision_row) < 0 and last_collision_row > -1:
         while last_collision_row >= 0 and len(tetromino) > 1:
            gameover = True
            last_collision_row = -1
            del tetromino[0]
            for y in range(len(tetromino)):
               for x in range(len(tetromino[y])):
                  if self.board[pos["y"] + y][pos["x"] + x] and tetromino[y][x] and y > last_collision_row:
                     last_collision_row = y
      return gameover

   def store(self, tetromino, pos):
      board = deepcopy(self.board)
      # try to convert using itertools
      for y in range(len(tetromino)):
         for x in range(len(tetromino[y])):
            if tetromino[y][x] and not board[y + pos["y"]][x + pos["x"]]:
               board[y + pos["y"]][x + pos["x"]] = tetromino[y][x]
      return board


   def compute_reward(self, lines, holes, height, bumpiness, gameover):
      # Parameters for the reward function
      a = -0.5
      b = -0.35
      c = -0.2
      # print([lines, holes, height, bumpiness])
      return -4 if gameover else (a*height)+lines**2+(b*holes)+(c*bumpiness) 

   def step(self, action, render=True, video=None):
      x, rotations = action
      self.current_pos = {'x': x, 'y': 0}
      for _ in range(rotations):
         self.tetromino = self.rotate(self.tetromino)
      while not self.check_collision(self.tetromino, self.current_pos):
         self.current_pos['y'] += 1
         if render:
            self.render(video)
      if self.truncate(self.tetromino, self.current_pos):
         self.gameover = True
      self.board = self.store(self.tetromino, self.current_pos)
      line_count, self.board = self.completed_lines(self.board)
      score = 1 + (line_count ** 2) * WITDH
      self.score += score
      self.count += 1
      self.lines += line_count
      
      if self.gameover:
         self.score -= 2
      else:
         self.new_piece()

      reward = self.compute_reward(
         self.lines, 
         self.hole_count(self.board), 
         sum(self.heights(self.board)), 
         self.bumpiness(self.board),
         self.gameover
         )

      return (self.next_states(), reward, self.gameover)

   def render(self, video=None):
      if not self.gameover:
         img = [self.tetromino_color[p] for row in self.current_board_state() for p in row]
      else:
         img = [self.tetromino_color[p] for row in self.board for p in row]
      img = np.array(img).reshape((HEIGHT, WITDH, 3)).astype(np.uint8)
      img = img[..., ::-1]
      img = Image.fromarray(img, 'RGB')

      img = img.resize((WITDH * BLKSIZE, HEIGHT * BLKSIZE), 0)
      img = np.array(img)
      img[[i * BLKSIZE for i in range(HEIGHT)], :, :] = 0
      img[:, [i * BLKSIZE for i in range(WITDH)], :] = 0
      img = np.concatenate((img, self.side_panel), axis=1)

      cv2.putText(img, "Score:", (WITDH * BLKSIZE + int(BLKSIZE / 2), BLKSIZE),
               fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, color=self.text_color)
      cv2.putText(img, str(self.score), (WITDH * BLKSIZE + int(BLKSIZE / 2), 2 * BLKSIZE),
               fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=self.text_color)
      cv2.putText(img, "Pieces:", (WITDH * BLKSIZE + int(BLKSIZE / 2), 4 * BLKSIZE),
               fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, color=self.text_color)
      cv2.putText(img, str(self.count), (WITDH * BLKSIZE + int(BLKSIZE / 2), 5 * BLKSIZE),
               fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=self.text_color)
      cv2.putText(img, "Lines:", (WITDH* BLKSIZE + int(BLKSIZE / 2), 7 * BLKSIZE),
               fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, color=self.text_color)
      cv2.putText(img, str(self.lines), (WITDH * BLKSIZE + int(BLKSIZE / 2), 8 * BLKSIZE),
               fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=self.text_color)

      if video:
         video.write(img)

      cv2.imshow("Deep Q-Learning Tetris", img)
      cv2.waitKey(1)





