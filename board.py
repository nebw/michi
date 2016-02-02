#!/usr/bin/python2

from collections import namedtuple
import random
import re

from config import *

#######################
# board string routines

def neighbors(c):
    """ generator of coordinates for all neighbors of c """
    return [c-1, c+1, c-W, c+W]

def diag_neighbors(c):
    """ generator of coordinates for all diagonal neighbors of c """
    return [c-W-1, c-W+1, c+W-1, c+W+1]


def board_put(board, c, p):
    return board[:c] + p + board[c+1:]


def floodfill(board, c):
    """ replace continuous-color area starting at c with special color #
"""
    # This is called so much that a bytearray is worthwhile...
    byteboard = bytearray(board)
    p = byteboard[c]
    byteboard[c] = ord('#')
    fringe = [c]
    while fringe:
        c = fringe.pop()
        for d in neighbors(c):
            if byteboard[d] == p:
                byteboard[d] = ord('#')
                fringe.append(d)
    return str(byteboard)


# Regex that matches various kind of points adjecent to '#' (floodfilled) points
contact_res = dict()
for p in ['.', 'x', 'X']:
    rp = '\\.' if p == '.' else p
    contact_res_src = ['#' + rp,  # p at right
                       rp + '#',  # p at left
                       '#' + '.'*(W-1) + rp,  # p below
                       rp + '.'*(W-1) + '#']  # p above
    contact_res[p] = re.compile('|'.join(contact_res_src), flags=re.DOTALL)

def contact(board, p):
    """ test if point of color p is adjecent to color # anywhere
    on the board; use in conjunction with floodfill for reachability """
    m = contact_res[p].search(board)
    if not m:
        return None
    return m.start() if m.group(0)[0] == p else m.end() - 1


def is_eyeish(board, c):
    """ test if c is inside a single-color diamond and return the diamond
    color or None; this could be an eye, but also a false one """
    eyecolor = None
    for d in neighbors(c):
        if board[d].isspace():
            continue
        if board[d] == '.':
            return None
        if eyecolor is None:
            eyecolor = board[d]
            othercolor = eyecolor.swapcase()
        elif board[d] == othercolor:
            return None
    return eyecolor

def is_eye(board, c):
    """ test if c is an eye and return its color or None """
    eyecolor = is_eyeish(board, c)
    if eyecolor is None:
        return None

    # Eye-like shape, but it could be a falsified eye
    falsecolor = eyecolor.swapcase()
    false_count = 0
    at_edge = False
    for d in diag_neighbors(c):
        if board[d].isspace():
            at_edge = True
        elif board[d] == falsecolor:
            false_count += 1
    if at_edge:
        false_count += 1
    if false_count >= 2:
        return None

    return eyecolor

class Position(namedtuple('Position', 'board cap n ko last last2 komi')):
    """ Implementation of simple Chinese Go rules;
    n is how many moves were played so far """

    def move(self, c):
        """ play as player X at the given coord c, return the new position """

        # Test for ko
        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        board = board_put(self.board, c, 'X')
        # Test for captures, and track ko
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
            board = fboard.replace('#', '.')  # capture the group
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None

        # Update the position and return
        return Position(board=board.swapcase(), cap=(self.cap[1], capX),
                        n=self.n + 1, ko=ko, last=c, last2=self.last, komi=self.komi)

    def pass_move(self):
        """ pass - i.e. return simply a flipped position """
        return Position(board=self.board.swapcase(), cap=(self.cap[1], self.cap[0]),
                        n=self.n + 1, ko=None, last=None, last2=self.last, komi=self.komi)

    def moves(self, i0):
        """ Generate a list of moves (includes false positives - suicide moves;
        does not include true-eye-filling moves), starting from a given board
        index (that can be used for randomization) """
        i = i0-1
        passes = 0
        while True:
            i = self.board.find('.', i+1)
            if passes > 0 and (i == -1 or i >= i0):
                break  # we have looked through the whole board
            elif i == -1:
                i = 0
                passes += 1
                continue  # go back and start from the beginning
            # Test for to-play player's one-point eye
            if is_eye(self.board, i) == 'X':
                continue
            yield i

    def last_moves_neighbors(self):
        """ generate a randomly shuffled list of points including and
        surrounding the last two moves (but with the last move having
        priority) """
        clist = []
        for c in self.last, self.last2:
            if c is None:  continue
            dlist = [c] + list(neighbors(c) + diag_neighbors(c))
            random.shuffle(dlist)
            clist += [d for d in dlist if d not in clist]
        return clist

    def score(self, owner_map=None):
        """ compute score for to-play player; this assumes a final position
        with all dead stones captured; if owner_map is passed, it is assumed
        to be an array of statistics with average owner at the end of the game
        (+1 black, -1 white) """
        board = self.board
        i = 0
        while True:
            i = self.board.find('.', i+1)
            if i == -1:
                break
            fboard = floodfill(board, i)
            # fboard is board with some continuous area of empty space replaced by #
            touches_X = contact(fboard, 'X') is not None
            touches_x = contact(fboard, 'x') is not None
            if touches_X and not touches_x:
                board = fboard.replace('#', 'X')
            elif touches_x and not touches_X:
                board = fboard.replace('#', 'x')
            else:
                board = fboard.replace('#', ':')  # seki, rare
            # now that area is replaced either by X, x or :
        komi = self.komi if self.n % 2 == 1 else -self.komi
        if owner_map is not None:
            for c in range(W*W):
                n = 1 if board[c] == 'X' else -1 if board[c] == 'x' else 0
                owner_map[c] += n * (1 if self.n % 2 == 0 else -1)
        return board.count('X') - board.count('x') + komi


def empty_position():
    """ Return an initial board position """
    return Position(board=empty, cap=(0, 0), n=0, ko=None, last=None, last2=None, komi=7.5)
