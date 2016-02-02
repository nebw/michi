#!/usr/bin/python2

from board import *
from config import *


###############
# go heuristics

def fix_atari(pos, c, singlept_ok=False, twolib_test=True, twolib_edgeonly=False):
    """ An atari/capture analysis routine that checks the group at c,
    determining whether (i) it is in atari (ii) if it can escape it,
    either by playing on its liberty or counter-capturing another group.

    N.B. this is maybe the most complicated part of the whole program (sadly);
    feel free to just TREAT IT AS A BLACK-BOX, it's not really that
    interesting!

    The return value is a tuple of (boolean, [coord..]), indicating whether
    the group is in atari and how to escape/capture (or [] if impossible).
    (Note that (False, [...]) is possible in case the group can be captured
    in a ladder - it is not in atari but some capture attack/defense moves
    are available.)

    singlept_ok means that we will not try to save one-point groups;
    twolib_test means that we will check for 2-liberty groups which are
    threatened by a ladder
    twolib_edgeonly means that we will check the 2-liberty groups only
    at the board edge, allowing check of the most common short ladders
    even in the playouts """

    def read_ladder_attack(pos, c, l1, l2):
        """ check if a capturable ladder is being pulled out at c and return
        a move that continues it in that case; expects its two liberties as
        l1, l2  (in fact, this is a general 2-lib capture exhaustive solver) """
        for l in [l1, l2]:
            pos_l = pos.move(l)
            if pos_l is None:
                continue
            # fix_atari() will recursively call read_ladder_attack() back;
            # however, ignore 2lib groups as we don't have time to chase them
            is_atari, atari_escape = fix_atari(pos_l, c, twolib_test=False)
            if is_atari and not atari_escape:
                return l
        return None

    fboard = floodfill(pos.board, c)
    group_size = fboard.count('#')
    if singlept_ok and group_size == 1:
        return (False, [])
    # Find a liberty
    l = contact(fboard, '.')
    # Ok, any other liberty?
    fboard = board_put(fboard, l, 'L')
    l2 = contact(fboard, '.')
    if l2 is not None:
        # At least two liberty group...
        if twolib_test and group_size > 1 \
           and (not twolib_edgeonly or line_height(l) == 0 and line_height(l2) == 0) \
           and contact(board_put(fboard, l2, 'L'), '.') is None:
            # Exactly two liberty group with more than one stone.  Check
            # that it cannot be caught in a working ladder; if it can,
            # that's as good as in atari, a capture threat.
            # (Almost - N/A for countercaptures.)
            ladder_attack = read_ladder_attack(pos, c, l, l2)
            if ladder_attack:
                return (False, [ladder_attack])
        return (False, [])

    # In atari! If it's the opponent's group, that's enough...
    if pos.board[c] == 'x':
        return (True, [l])

    solutions = []

    # Before thinking about defense, what about counter-capturing
    # a neighboring group?
    ccboard = fboard
    while True:
        othergroup = contact(ccboard, 'x')
        if othergroup is None:
            break
        a, ccls = fix_atari(pos, othergroup, twolib_test=False)
        if a and ccls:
            solutions += ccls
        # XXX: floodfill is better for big groups
        ccboard = board_put(ccboard, othergroup, '%')

    # We are escaping.  Will playing our last liberty gain
    # at least two liberties?  Re-floodfill to account for connecting
    escpos = pos.move(l)
    if escpos is None:
        return (True, solutions)  # oops, suicidal move
    fboard = floodfill(escpos.board, l)
    l_new = contact(fboard, '.')
    fboard = board_put(fboard, l_new, 'L')
    l_new_2 = contact(fboard, '.')
    if l_new_2 is not None:
        # Good, there is still some liberty remaining - but if it's
        # just the two, check that we are not caught in a ladder...
        # (Except that we don't care if we already have some alternative
        # escape routes!)

        if solutions or not (contact(board_put(fboard, l_new_2, 'L'), '.') is None
                             and read_ladder_attack(escpos, l, l_new, l_new_2) is not None):
            solutions.append(l)

    return (True, solutions)


def cfg_distances(board, c):
    """ return a board map listing common fate graph distances from
    a given point - this corresponds to the concept of locality while
    contracting groups to single points """
    cfg_map = W*W*[-1]
    cfg_map[c] = 0

    # flood-fill like mechanics
    fringe = [c]
    while fringe:
        c = fringe.pop()
        for d in neighbors(c):
            if board[d].isspace() or 0 <= cfg_map[d] <= cfg_map[c]:
                continue
            cfg_before = cfg_map[d]
            if board[d] != '.' and board[d] == board[c]:
                cfg_map[d] = cfg_map[c]
            else:
                cfg_map[d] = cfg_map[c] + 1
            if cfg_before < 0 or cfg_before > cfg_map[d]:
                fringe.append(d)
    return cfg_map

def line_height(c):
    """ Return the line number above nearest board edge """
    row, col = divmod(c - (W+1), W)
    return min(row, col, N-1-row, N-1-col)


def empty_area(board, c, dist=3):
    """ Check whether there are any stones in Manhattan distance up
    to dist """
    for d in neighbors(c):
        if board[d] in 'Xx':
            return False
        elif board[d] == '.' and dist > 1 and not empty_area(board, d, dist-1):
            return False
    return True


# 3x3 pattern routines (those patterns stored in pat3src above)

def pat3_expand(pat):
    """ All possible neighborhood configurations matching a given pattern;
    used just for a combinatoric explosion when loading them in an
    in-memory set. """
    def pat_rot90(p):
        return [p[2][0] + p[1][0] + p[0][0], p[2][1] + p[1][1] + p[0][1], p[2][2] + p[1][2] + p[0][2]]
    def pat_vertflip(p):
        return [p[2], p[1], p[0]]
    def pat_horizflip(p):
        return [l[::-1] for l in p]
    def pat_swapcolors(p):
        return [l.replace('X', 'Z').replace('x', 'z').replace('O', 'X').replace('o', 'x').replace('Z', 'O').replace('z', 'o') for l in p]
    def pat_wildexp(p, c, to):
        i = p.find(c)
        if i == -1:
            return [p]
        return reduce(lambda a, b: a + b, [pat_wildexp(p[:i] + t + p[i+1:], c, to) for t in to])
    def pat_wildcards(pat):
        return [p for p in pat_wildexp(pat, '?', list('.XO '))
                  for p in pat_wildexp(p, 'x', list('.O '))
                  for p in pat_wildexp(p, 'o', list('.X '))]
    return [p for p in [pat, pat_rot90(pat)]
              for p in [p, pat_vertflip(p)]
              for p in [p, pat_horizflip(p)]
              for p in [p, pat_swapcolors(p)]
              for p in pat_wildcards(''.join(p))]

pat3set = set([p.replace('O', 'x') for p in pat3src for p in pat3_expand(p)])

def neighborhood_33(board, c):
    """ return a string containing the 9 points forming 3x3 square around
    a certain move candidate """
    return (board[c-W-1 : c-W+2] + board[c-1 : c+2] + board[c+W-1 : c+W+2]).replace('\n', ' ')


# large-scale pattern routines (those patterns living in patterns.{spat,prob} files)

# are you curious how these patterns look in practice? get
# https://github.com/pasky/pachi/blob/master/tools/pattern_spatial_show.pl
# and try e.g. ./pattern_spatial_show.pl 71

spat_patterndict = dict()  # hash(neighborhood_gridcular()) -> spatial id
def load_spat_patterndict(f):
    """ load dictionary of positions, translating them to numeric ids """
    for line in f:
        # line: 71 6 ..X.X..OO.O..........#X...... 33408f5e 188e9d3e 2166befe aa8ac9e 127e583e 1282462e 5e3d7fe 51fc9ee
        if line.startswith('#'):
            continue
        neighborhood = line.split()[2].replace('#', ' ').replace('O', 'x')
        spat_patterndict[hash(neighborhood)] = int(line.split()[0])

large_patterns = dict()  # spatial id -> probability
def load_large_patterns(f):
    """ dictionary of numeric pattern ids, translating them to probabilities
    that a move matching such move will be played when it is available """
    # The pattern file contains other features like capture, selfatari too;
    # we ignore them for now
    for line in f:
        # line: 0.004 14 3842 (capture:17 border:0 s:784)
        p = float(line.split()[0])
        m = re.search('s:(\d+)', line)
        if m is not None:
            s = int(m.groups()[0])
            large_patterns[s] = p



def neighborhood_gridcular(board, c):
    """ Yield progressively wider-diameter gridcular board neighborhood
    stone configuration strings, in all possible rotations """
    # Each rotations element is (xyindex, xymultiplier)
    rotations = [((0,1),(1,1)), ((0,1),(-1,1)), ((0,1),(1,-1)), ((0,1),(-1,-1)),
                 ((1,0),(1,1)), ((1,0),(-1,1)), ((1,0),(1,-1)), ((1,0),(-1,-1))]
    neighborhood = ['' for i in range(len(rotations))]
    wboard = board.replace('\n', ' ')
    for dseq in pat_gridcular_seq:
        for ri in range(len(rotations)):
            r = rotations[ri]
            for o in dseq:
                y, x = divmod(c - (W+1), W)
                y += o[r[0][0]]*r[1][0]
                x += o[r[0][1]]*r[1][1]
                if y >= 0 and y < N and x >= 0 and x < N:
                    neighborhood[ri] += wboard[(y+1)*W + x+1]
                else:
                    neighborhood[ri] += ' '
            yield neighborhood[ri]


def large_pattern_probability(board, c):
    """ return probability of large-scale pattern at coordinate c.
    Multiple progressively wider patterns may match a single coordinate,
    we consider the largest one. """
    probability = None
    matched_len = 0
    non_matched_len = 0
    for n in neighborhood_gridcular(board, c):
        sp_i = spat_patterndict.get(hash(n))
        prob = large_patterns.get(sp_i) if sp_i is not None else None
        if prob is not None:
            probability = prob
            matched_len = len(n)
        elif matched_len < non_matched_len < len(n):
            # stop when we did not match any pattern with a certain
            # diameter - it ain't going to get any better!
            break
        else:
            non_matched_len = len(n)
    return probability

