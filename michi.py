#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
#
# (c) Petr Baudis <pasky@ucw.cz>  2015
# MIT licence (i.e. almost public domain)
#
# A minimalistic Go-playing engine attempting to strike a balance between
# brevity, educational value and strength.  It can beat GNUGo on 13x13 board
# on a modest 4-thread laptop.
#
# When benchmarking, note that at the beginning of the first move the program
# runs much slower because pypy is JIT compiling on the background!
#
# To start reading the code, begin either:
# * Bottom up, by looking at the goban implementation - starting with
#   the 'empty' definition below and Position.move() method.
# * In the middle, by looking at the Monte Carlo playout implementation,
#   starting with the mcplayout() function.
# * Top down, by looking at the MCTS implementation, starting with the
#   tree_search() function.  It can look a little confusing due to the
#   parallelization, but really is just a loop of tree_descend(),
#   mcplayout() and tree_update() round and round.
# It may be better to jump around a bit instead of just reading straight
# from start to end.

from __future__ import print_function
import time

from board import *
from config import *
from mcts import *
from util import *


###################
# user interface(s)

# various main programs

def mcbenchmark(n):
    """ run n Monte-Carlo playouts from empty position, return avg. score """
    sumscore = 0
    for i in range(0, n):
        sumscore += mcplayout(empty_position(), W*W*[0])[0]
    return float(sumscore) / n


def game_io(computer_black=False):
    """ A simple minimalistic text mode UI. """

    tree = TreeNode(pos=empty_position())
    tree.expand()
    owner_map = W*W*[0]
    while True:
        if not (tree.pos.n == 0 and computer_black):
            print_pos(tree.pos, sys.stdout, owner_map)

            sc = raw_input("Your move: ")
            c = parse_coord(sc)
            if c is not None:
                # Not a pass
                if tree.pos.board[c] != '.':
                    print('Bad move (not empty point)')
                    continue

                # Find the next node in the game tree and proceed there
                nodes = filter(lambda n: n.pos.last == c, tree.children)
                if not nodes:
                    print('Bad move (rule violation)')
                    continue
                tree = nodes[0]

            else:
                # Pass move
                if tree.children[0].pos.last is None:
                    tree = tree.children[0]
                else:
                    tree = TreeNode(pos=tree.pos.pass_move())

            print_pos(tree.pos)

        owner_map = W*W*[0]
        tree = tree_search(tree, N_SIMS, owner_map)
        if tree.pos.last is None and tree.pos.last2 is None:
            score = tree.pos.score()
            if tree.pos.n % 2:
                score = -score
            print('Game over, score: B%+.1f' % (score,))
            break
        if float(tree.w)/tree.v < RESIGN_THRES:
            print('I resign.')
            break
    print('Thank you for the game!')


def gtp_io():
    """ GTP interface for our program.  We can play only on the board size
    which is configured (N), and we ignore color information and assume
    alternating play! """
    known_commands = ['boardsize', 'clear_board', 'komi', 'play', 'genmove',
                      'final_score', 'quit', 'name', 'version', 'known_command',
                      'list_commands', 'protocol_version', 'tsdebug']

    tree = TreeNode(pos=empty_position())
    tree.expand()

    while True:
        try:
            line = raw_input().strip()
        except EOFError:
            break
        if line == '':
            continue
        command = [s.lower() for s in line.split()]
        if re.match('\d+', command[0]):
            cmdid = command[0]
            command = command[1:]
        else:
            cmdid = ''
        owner_map = W*W*[0]
        ret = ''
        if command[0] == "boardsize":
            if int(command[1]) != N:
                print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], N), file=sys.stderr)
        elif command[0] == "clear_board":
            tree = TreeNode(pos=empty_position())
            tree.expand()
        elif command[0] == "komi":
            # XXX: can we do this nicer?!
            tree.pos = Position(board=tree.pos.board, cap=(tree.pos.cap[0], tree.pos.cap[1]),
                                n=tree.pos.n, ko=tree.pos.ko, last=tree.pos.last, last2=tree.pos.last2,
                                komi=float(command[1]))
        elif command[0] == "play":
            c = parse_coord(command[2])
            if c is not None:
                # Find the next node in the game tree and proceed there
                if tree.children is not None and filter(lambda n: n.pos.last == c, tree.children):
                    tree = filter(lambda n: n.pos.last == c, tree.children)[0]
                else:
                    # Several play commands in row, eye-filling move, etc.
                    tree = TreeNode(pos=tree.pos.move(c))

            else:
                # Pass move
                if tree.children[0].pos.last is None:
                    tree = tree.children[0]
                else:
                    tree = TreeNode(pos=tree.pos.pass_move())
        elif command[0] == "genmove":
            tree = tree_search(tree, N_SIMS, owner_map)
            if tree.pos.last is None:
                ret = 'pass'
            elif float(tree.w)/tree.v < RESIGN_THRES:
                ret = 'resign'
            else:
                ret = str_coord(tree.pos.last)
        elif command[0] == "final_score":
            score = tree.pos.score()
            if tree.pos.n % 2:
                score = -score
            if score == 0:
                ret = '0'
            elif score > 0:
                ret = 'B+%.1f' % (score,)
            elif score < 0:
                ret = 'W+%.1f' % (-score,)
        elif command[0] == "name":
            ret = 'michi'
        elif command[0] == "version":
            ret = 'simple go program demo'
        elif command[0] == "tsdebug":
            print_pos(tree_search(tree, N_SIMS, W*W*[0], disp=True))
        elif command[0] == "list_commands":
            ret = '\n'.join(known_commands)
        elif command[0] == "known_command":
            ret = 'true' if command[1] in known_commands else 'false'
        elif command[0] == "protocol_version":
            ret = '2'
        elif command[0] == "quit":
            print('=%s \n\n' % (cmdid,), end='')
            break
        else:
            print('Warning: Ignoring unknown command - %s' % (line,), file=sys.stderr)
            ret = None

        print_pos(tree.pos, sys.stderr, owner_map)
        if ret is not None:
            print('=%s %s\n\n' % (cmdid, ret,), end='')
        else:
            print('?%s ???\n\n' % (cmdid,), end='')
        sys.stdout.flush()


if __name__ == "__main__":
    try:
        with open(spat_patterndict_file) as f:
            print('Loading pattern spatial dictionary...', file=sys.stderr)
            load_spat_patterndict(f)
        with open(large_patterns_file) as f:
            print('Loading large patterns...', file=sys.stderr)
            load_large_patterns(f)
        print('Done.', file=sys.stderr)
    except IOError as e:
        print('Warning: Cannot load pattern files: %s; will be much weaker, consider lowering EXPAND_VISITS 5->2' % (e,), file=sys.stderr)
    if len(sys.argv) < 2:
        # Default action
        game_io()
    elif sys.argv[1] == "white":
        game_io(computer_black=True)
    elif sys.argv[1] == "gtp":
        gtp_io()
    elif sys.argv[1] == "mcdebug":
        print(mcplayout(empty_position(), W*W*[0], disp=True)[0])
    elif sys.argv[1] == "mcbenchmark":
        print(mcbenchmark(20))
    elif sys.argv[1] == "tsbenchmark":
        t_start = time.time()
        print_pos(tree_search(TreeNode(pos=empty_position()), N_SIMS, W*W*[0], disp=False).pos)
        print('Tree search with %d playouts took %.3fs with %d threads; speed is %.3f playouts/thread/s' %
              (N_SIMS, time.time() - t_start, multiprocessing.cpu_count(),
               N_SIMS / ((time.time() - t_start) * multiprocessing.cpu_count())))
    elif sys.argv[1] == "tsdebug":
        print_pos(tree_search(TreeNode(pos=empty_position()), N_SIMS, W*W*[0], disp=True).pos)
    else:
        print('Unknown action', file=sys.stderr)
