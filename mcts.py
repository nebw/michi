#!/usr/bin/python2

from __future__ import print_function
import math
import multiprocessing
from multiprocessing.pool import Pool

from board import *
from config import *
from heuristics import *
from michi import *

###########################
# montecarlo playout policy

def gen_playout_moves(pos, heuristic_set, probs={'capture': 1, 'pat3': 1}, expensive_ok=False):
    """ Yield candidate next moves in the order of preference; this is one
    of the main places where heuristics dwell, try adding more!

    heuristic_set is the set of coordinates considered for applying heuristics;
    this is the immediate neighborhood of last two moves in the playout, but
    the whole board while prioring the tree. """

    # Check whether any local group is in atari and fill that liberty
    # print('local moves', [str_coord(c) for c in heuristic_set], file=sys.stderr)
    if random.random() <= probs['capture']:
        already_suggested = set()
        for c in heuristic_set:
            if pos.board[c] in 'Xx':
                in_atari, ds = fix_atari(pos, c, twolib_edgeonly=not expensive_ok)
                random.shuffle(ds)
                for d in ds:
                    if d not in already_suggested:
                        yield (d, 'capture '+str(c))
                        already_suggested.add(d)

    # Try to apply a 3x3 pattern on the local neighborhood
    if random.random() <= probs['pat3']:
        already_suggested = set()
        for c in heuristic_set:
            if pos.board[c] == '.' and c not in already_suggested and neighborhood_33(pos.board, c) in pat3set:
                yield (c, 'pat3')
                already_suggested.add(c)

    # Try *all* available moves, but starting from a random point
    # (in other words, suggest a random move)
    x, y = random.randint(1, N), random.randint(1, N)
    for c in pos.moves(y*W + x):
        yield (c, 'random')

def mcplayout(pos, amaf_map, disp=False):
    """ Start a Monte Carlo playout from a given position,
    return score for to-play player at the starting position;
    amaf_map is board-sized scratchpad recording who played at a given
    position first """
    if disp:  print('** SIMULATION **', file=sys.stderr)
    start_n = pos.n
    passes = 0
    while passes < 2 and pos.n < MAX_GAME_LEN:
        if disp:  print_pos(pos)

        pos2 = None
        # We simply try the moves our heuristics generate, in a particular
        # order, but not with 100% probability; this is on the border between
        # "rule-based playouts" and "probability distribution playouts".
        for c, kind in gen_playout_moves(pos, pos.last_moves_neighbors(), PROB_HEURISTIC):
            if disp and kind != 'random':
                print('move suggestion', str_coord(c), kind, file=sys.stderr)
            pos2 = pos.move(c)
            if pos2 is None:
                continue
            # check if the suggested move did not turn out to be a self-atari
            if random.random() <= (PROB_RSAREJECT if kind == 'random' else PROB_SSAREJECT):
                in_atari, ds = fix_atari(pos2, c, singlept_ok=True, twolib_edgeonly=True)
                if ds:
                    if disp:  print('rejecting self-atari move', str_coord(c), file=sys.stderr)
                    pos2 = None
                    continue
            if amaf_map[c] == 0:  # Mark the coordinate with 1 for black
                amaf_map[c] = 1 if pos.n % 2 == 0 else -1
            break
        if pos2 is None:  # no valid moves, pass
            pos = pos.pass_move()
            passes += 1
            continue
        passes = 0
        pos = pos2

    owner_map = W*W*[0]
    score = pos.score(owner_map)
    if disp:  print('** SCORE B%+.1f **' % (score if pos.n % 2 == 0 else -score), file=sys.stderr)
    if start_n % 2 != pos.n % 2:
        score = -score
    return score, amaf_map, owner_map


########################
# montecarlo tree search

class TreeNode():
    """ Monte-Carlo tree node;
    v is #visits, w is #wins for to-play (expected reward is w/v)
    pv, pw are prior values (node value = w/v + pw/pv)
    av, aw are amaf values ("all moves as first", used for the RAVE tree policy)
    children is None for leaf nodes """
    def __init__(self, pos):
        self.pos = pos
        self.v = 0
        self.w = 0
        self.pv = PRIOR_EVEN
        self.pw = PRIOR_EVEN/2
        self.av = 0
        self.aw = 0
        self.children = None

    def expand(self):
        """ add and initialize children to a leaf node """
        cfg_map = cfg_distances(self.pos.board, self.pos.last) if self.pos.last is not None else None
        self.children = []
        childset = dict()
        # Use playout generator to generate children and initialize them
        # with some priors to bias search towards more sensible moves.
        # Note that there can be many ways to incorporate the priors in
        # next node selection (progressive bias, progressive widening, ...).
        for c, kind in gen_playout_moves(self.pos, range(N, (N+1)*W), expensive_ok=True):
            pos2 = self.pos.move(c)
            if pos2 is None:
                continue
            # gen_playout_moves() will generate duplicate suggestions
            # if a move is yielded by multiple heuristics
            try:
                node = childset[pos2.last]
            except KeyError:
                node = TreeNode(pos2)
                self.children.append(node)
                childset[pos2.last] = node

            if kind.startswith('capture'):
                # Check how big group we are capturing; coord of the group is
                # second word in the ``kind`` string
                if floodfill(self.pos.board, int(kind.split()[1])).count('#') > 1:
                    node.pv += PRIOR_CAPTURE_MANY
                    node.pw += PRIOR_CAPTURE_MANY
                else:
                    node.pv += PRIOR_CAPTURE_ONE
                    node.pw += PRIOR_CAPTURE_ONE
            elif kind == 'pat3':
                node.pv += PRIOR_PAT3
                node.pw += PRIOR_PAT3

        # Second pass setting priors, considering each move just once now
        for node in self.children:
            c = node.pos.last

            if cfg_map is not None and cfg_map[c]-1 < len(PRIOR_CFG):
                node.pv += PRIOR_CFG[cfg_map[c]-1]
                node.pw += PRIOR_CFG[cfg_map[c]-1]

            height = line_height(c)  # 0-indexed
            if height <= 2 and empty_area(self.pos.board, c):
                # No stones around; negative prior for 1st + 2nd line, positive
                # for 3rd line; sanitizes opening and invasions
                if height <= 1:
                    node.pv += PRIOR_EMPTYAREA
                    node.pw += 0
                if height == 2:
                    node.pv += PRIOR_EMPTYAREA
                    node.pw += PRIOR_EMPTYAREA

            in_atari, ds = fix_atari(node.pos, c, singlept_ok=True)
            if ds:
                node.pv += PRIOR_SELFATARI
                node.pw += 0  # negative prior

            patternprob = large_pattern_probability(self.pos.board, c)
            if patternprob is not None and patternprob > 0.001:
                pattern_prior = math.sqrt(patternprob)  # tone up
                node.pv += pattern_prior * PRIOR_LARGEPATTERN
                node.pw += pattern_prior * PRIOR_LARGEPATTERN

        if not self.children:
            # No possible moves, add a pass move
            self.children.append(TreeNode(self.pos.pass_move()))

    def rave_urgency(self):
        v = self.v + self.pv
        expectation = float(self.w+self.pw) / v
        if self.av == 0:
            return expectation
        rave_expectation = float(self.aw) / self.av
        beta = self.av / (self.av + v + float(v) * self.av / RAVE_EQUIV)
        return beta * rave_expectation + (1-beta) * expectation

    def winrate(self):
        return float(self.w) / self.v if self.v > 0 else float('nan')

    def best_move(self):
        """ best move is the most simulated one """
        return max(self.children, key=lambda node: node.v) if self.children is not None else None


def tree_descend(tree, amaf_map, disp=False):
    """ Descend through the tree to a leaf """
    tree.v += 1
    nodes = [tree]
    passes = 0
    while nodes[-1].children is not None and passes < 2:
        if disp:  print_pos(nodes[-1].pos)

        # Pick the most urgent child
        children = list(nodes[-1].children)
        if disp:
            for c in children:
                dump_subtree(c, recurse=False)
        random.shuffle(children)  # randomize the max in case of equal urgency
        node = max(children, key=lambda node: node.rave_urgency())
        nodes.append(node)

        if disp:  print('chosen %s' % (str_coord(node.pos.last),), file=sys.stderr)
        if node.pos.last is None:
            passes += 1
        else:
            passes = 0
            if amaf_map[node.pos.last] == 0:  # Mark the coordinate with 1 for black
                amaf_map[node.pos.last] = 1 if nodes[-2].pos.n % 2 == 0 else -1

        # updating visits on the way *down* represents "virtual loss", relevant for parallelization
        node.v += 1
        if node.children is None and node.v >= EXPAND_VISITS:
            node.expand()

    return nodes


def tree_update(nodes, amaf_map, score, disp=False):
    """ Store simulation result in the tree (@nodes is the tree path) """
    for node in reversed(nodes):
        if disp:  print('updating', str_coord(node.pos.last), score < 0, file=sys.stderr)
        node.w += score < 0  # score is for to-play, node statistics for just-played
        # Update the node children AMAF stats with moves we made
        # with their color
        amaf_map_value = 1 if node.pos.n % 2 == 0 else -1
        if node.children is not None:
            for child in node.children:
                if child.pos.last is None:
                    continue
                if amaf_map[child.pos.last] == amaf_map_value:
                    if disp:  print('  AMAF updating', str_coord(child.pos.last), score > 0, file=sys.stderr)
                    child.aw += score > 0  # reversed perspective
                    child.av += 1
        score = -score


worker_pool = None

def tree_search(tree, n, owner_map, disp=False):
    """ Perform MCTS search from a given position for a given #iterations """
    # Initialize root node
    if tree.children is None:
        tree.expand()

    # We could simply run tree_descend(), mcplayout(), tree_update()
    # sequentially in a loop.  This is essentially what the code below
    # does, if it seems confusing!

    # However, we also have an easy (though not optimal) way to parallelize
    # by distributing the mcplayout() calls to other processes using the
    # multiprocessing Python module.  mcplayout() consumes maybe more than
    # 90% CPU, especially on larger boards.  (Except that with large patterns,
    # expand() in the tree descent phase may be quite expensive - we can tune
    # that tradeoff by adjusting the EXPAND_VISITS constant.)

    n_workers = multiprocessing.cpu_count() if not disp else 1  # set to 1 when debugging
    global worker_pool
    if worker_pool is None:
        worker_pool = Pool(processes=n_workers)
    outgoing = []  # positions waiting for a playout
    incoming = []  # positions that finished evaluation
    ongoing = []  # currently ongoing playout jobs
    i = 0
    while i < n:
        if not outgoing and not (disp and ongoing):
            # Descend the tree so that we have something ready when a worker
            # stops being busy
            amaf_map = W*W*[0]
            nodes = tree_descend(tree, amaf_map, disp=disp)
            outgoing.append((nodes, amaf_map))

        if len(ongoing) >= n_workers:
            # Too many playouts running? Wait a bit...
            ongoing[0][0].wait(0.01 / n_workers)
        else:
            i += 1
            if i > 0 and i % REPORT_PERIOD == 0:
                print_tree_summary(tree, i, f=sys.stderr)

            # Issue an mcplayout job to the worker pool
            nodes, amaf_map = outgoing.pop()
            ongoing.append((worker_pool.apply_async(mcplayout, (nodes[-1].pos, amaf_map, disp)), nodes))

        # Anything to store in the tree?  (We do this step out-of-order
        # picking up data from the previous round so that we don't stall
        # ready workers while we update the tree.)
        while incoming:
            score, amaf_map, owner_map_one, nodes = incoming.pop()
            tree_update(nodes, amaf_map, score, disp=disp)
            for c in range(W*W):
                owner_map[c] += owner_map_one[c]

        # Any playouts are finished yet?
        for job, nodes in ongoing:
            if not job.ready():
                continue
            # Yes! Queue them up for storing in the tree.
            score, amaf_map, owner_map_one = job.get()
            incoming.append((score, amaf_map, owner_map_one, nodes))
            ongoing.remove((job, nodes))

        # Early stop test
        best_wr = tree.best_move().winrate()
        if i > n*0.05 and best_wr > FASTPLAY5_THRES or i > n*0.2 and best_wr > FASTPLAY20_THRES:
            break

    for c in range(W*W):
        owner_map[c] = float(owner_map[c]) / i
    dump_subtree(tree)
    print_tree_summary(tree, i, f=sys.stderr)
    return tree.best_move()

