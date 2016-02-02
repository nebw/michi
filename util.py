#!/usr/bin/python2

from __future__ import print_function
from itertools import count
import sys

from config import *

# utility routines

def print_pos(pos, f=sys.stderr, owner_map=None):
    """ print visualization of the given board position, optionally also
    including an owner map statistic (probability of that area of board
    eventually becoming black/white) """
    if pos.n % 2 == 0:  # to-play is black
        board = pos.board.replace('x', 'O')
        Xcap, Ocap = pos.cap
    else:  # to-play is white
        board = pos.board.replace('X', 'O').replace('x', 'X')
        Ocap, Xcap = pos.cap
    print('Move: %-3d   Black: %d caps   White: %d caps  Komi: %.1f' % (pos.n, Xcap, Ocap, pos.komi), file=f)
    pretty_board = ' '.join(board.rstrip()) + ' '
    if pos.last is not None:
        pretty_board = pretty_board[:pos.last*2-1] + '(' + board[pos.last] + ')' + pretty_board[pos.last*2+2:]
    rowcounter = count()
    pretty_board = [' %-02d%s' % (N-i, row[2:]) for row, i in zip(pretty_board.split("\n")[1:], rowcounter)]
    if owner_map is not None:
        pretty_ownermap = ''
        for c in range(W*W):
            if board[c].isspace():
                pretty_ownermap += board[c]
            elif owner_map[c] > 0.6:
                pretty_ownermap += 'X'
            elif owner_map[c] > 0.3:
                pretty_ownermap += 'x'
            elif owner_map[c] < -0.6:
                pretty_ownermap += 'O'
            elif owner_map[c] < -0.3:
                pretty_ownermap += 'o'
            else:
                pretty_ownermap += '.'
        pretty_ownermap = ' '.join(pretty_ownermap.rstrip())
        pretty_board = ['%s   %s' % (brow, orow[2:]) for brow, orow in zip(pretty_board, pretty_ownermap.split("\n")[1:])]
    print("\n".join(pretty_board), file=f)
    print('    ' + ' '.join(colstr[:N]), file=f)
    print('', file=f)

def dump_subtree(node, thres=N_SIMS/50, indent=0, f=sys.stderr, recurse=True):
    """ print this node and all its children with v >= thres. """
    print("%s+- %s %.3f (%d/%d, prior %d/%d, rave %d/%d=%.3f, urgency %.3f)" %
          (indent*' ', str_coord(node.pos.last), node.winrate(),
           node.w, node.v, node.pw, node.pv, node.aw, node.av,
           float(node.aw)/node.av if node.av > 0 else float('nan'),
           node.rave_urgency()), file=f)
    if not recurse:
        return
    for child in sorted(node.children, key=lambda n: n.v, reverse=True):
        if child.v >= thres:
            dump_subtree(child, thres=thres, indent=indent+3, f=f)


def print_tree_summary(tree, sims, f=sys.stderr):
    best_nodes = sorted(tree.children, key=lambda n: n.v, reverse=True)[:5]
    best_seq = []
    node = tree
    while node is not None:
        best_seq.append(node.pos.last)
        node = node.best_move()
    print('[%4d] winrate %.3f | seq %s | can %s' %
          (sims, best_nodes[0].winrate(), ' '.join([str_coord(c) for c in best_seq[1:6]]),
           ' '.join(['%s(%.3f)' % (str_coord(n.pos.last), n.winrate()) for n in best_nodes])), file=f)


def parse_coord(s):
    if s == 'pass':
        return None
    return W+1 + (N - int(s[1:])) * W + colstr.index(s[0].upper())


def str_coord(c):
    if c is None:
        return 'pass'
    row, col = divmod(c - (W+1), W)
    return '%c%d' % (colstr[col], N - row)
