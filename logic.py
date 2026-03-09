import math
import time
import heapq
from collections import deque

# =================================================================
#  OPPONENT PROFILER STATE
# =================================================================

profile_state: dict = {}

_PROFILE_MIN_TURNS  = 15
_INTERACTION_RADIUS = 6


def _get_or_init_profile(game_id: str, snake_id: str) -> dict:
    if game_id not in profile_state:
        profile_state[game_id] = {}
    game = profile_state[game_id]
    if snake_id not in game:
        game[snake_id] = {
            'turn_count':       0,
            'prev_head':        None,
            'prev_food_dist':   None,
            'prev_thunga_dist': None,
            'vacuum_count':     0,
            'wall_count':       0,
            'aggro_count':      0,
            'center_count':     0,
            'archetype':        None,
            'confidence':       0.0,
        }
    return game[snake_id]


def _classify_archetype(p: dict):
    n = p['turn_count']
    if n < _PROFILE_MIN_TURNS:
        return (None, 0.0)

    v = p['vacuum_count'] / n
    w = p['wall_count']   / n
    a = p['aggro_count']  / n
    c = p['center_count'] / n

    if v > 0.85: return ('vacuum',       v)
    if w > 0.80: return ('camper',       w)
    if a > 0.60: return ('aggressor',    a)
    if c > 0.75: return ('area_control', c)

    max_ratio = max(v, w, a, c)
    if max_ratio < 0.50:
        return ('erratic', 1.0 - max_ratio)

    return (None, 0.0)


def update_profiles(game_state: dict, start_time: float = 0.0) -> None:
    game_id   = game_state['game']['id']
    board     = game_state['board']
    bw        = board['width']
    bh        = board['height']
    cx        = bw // 2
    cy        = bh // 2
    my_snake  = game_state['you']
    my_head   = my_snake['head']
    my_length = my_snake['length']
    food_list = board['food']

    for snake in board['snakes']:
        if snake['id'] == my_snake['id']:
            continue

        if time.perf_counter() - start_time > 0.050:
            break

        p  = _get_or_init_profile(game_id, snake['id'])
        hx = snake['head']['x']
        hy = snake['head']['y']

        cur_food_dist = min(
            (abs(hx - f['x']) + abs(hy - f['y']) for f in food_list),
            default=999
        )
        cur_thunga_dist = abs(hx - my_head['x']) + abs(hy - my_head['y'])

        if p['prev_head'] is not None:
            moved_toward_food   = (cur_food_dist   < p['prev_food_dist'])
            moved_toward_thunga = (cur_thunga_dist < p['prev_thunga_dist'])
            is_larger           = (snake['length'] > my_length)

            if moved_toward_food:
                p['vacuum_count'] += 1

            if hx == 0 or hx == bw - 1 or hy == 0 or hy == bh - 1:
                p['wall_count'] += 1

            if moved_toward_thunga and not moved_toward_food and is_larger:
                p['aggro_count'] += 1

            if (cx - 2 <= hx <= cx + 2) and (cy - 2 <= hy <= cy + 2):
                p['center_count'] += 1

            p['turn_count'] += 1

        p['prev_head']        = {'x': hx, 'y': hy}
        p['prev_food_dist']   = cur_food_dist
        p['prev_thunga_dist'] = cur_thunga_dist

        archetype, confidence = _classify_archetype(p)
        p['archetype']        = archetype
        p['confidence']       = confidence


def get_game_profiles(game_state: dict) -> dict:
    return profile_state.get(game_state['game']['id'], {})


# =================================================================
#  ENTRY POINT
# =================================================================

def choose_move(data):
    start_time = time.perf_counter()
    return choose_best_move(data, start_time=start_time)


# =================================================================
#  BOARD VISUALIZER
# =================================================================

def print_board(game_state):
    board      = game_state['board']
    my_snake   = game_state['you']
    width      = board['width']
    height     = board['height']

    grid = [["." for _ in range(width)] for _ in range(height)]

    for food in board['food']:
        grid[food['y']][food['x']] = "F"

    for snake in board['snakes']:
        if snake['id'] == my_snake['id']:
            continue
        for part in snake['body']:
            grid[part['y']][part['x']] = "e"
        grid[snake['head']['y']][snake['head']['x']] = "E"

    for part in my_snake['body']:
        grid[part['y']][part['x']] = "s"
    grid[my_snake['head']['y']][my_snake['head']['x']] = "S"

    print("\n" + "=" * (width * 2))
    for row in reversed(grid):
        print(" ".join(row))
    print("=" * (width * 2))
    print(f"  Turn: {game_state['turn']}  |  Health: {my_snake['health']}  |  Length: {my_snake['length']}  |  Snakes alive: {len(board['snakes'])}")
    print("=" * (width * 2))
    print("  S=you  s=body  E=enemy  e=enemy body  F=food")


# =================================================================
#  MAIN DECISION FUNCTION
# =================================================================

DEBUG = False


def choose_best_move(game_state, start_time: float = 0.0):

    print_board(game_state)

    update_profiles(game_state, start_time=start_time)
    game_profiles = get_game_profiles(game_state)

    safe_moves = get_safe_moves(game_state)
    in_risky_fallback = False

    if not safe_moves:
        return "up"

    all_moves = ["up", "down", "left", "right"]
    my_head   = game_state['you']['head']
    board_w   = game_state['board']['width']
    board_h   = game_state['board']['height']
    my_length = game_state['you']['length']

    occupied = set()
    for snake in game_state['board']['snakes']:
        tail_stays = _tail_will_stay(snake, game_state['board'])
        for i, part in enumerate(snake['body']):
            if not tail_stays and i == len(snake['body']) - 1:
                continue
            occupied.add((part['x'], part['y']))

    danger_squares = set()
    for snake in game_state['board']['snakes']:
        if snake['id'] == game_state['you']['id']:
            continue
        if snake['length'] >= my_length:
            for move in all_moves:
                nc = get_next_coord(snake['head'], move)
                nx, ny = nc['x'], nc['y']
                if 0 <= nx < board_w and 0 <= ny < board_h:
                    danger_squares.add((nx, ny))

    truly_safe = []
    for move in all_moves:
        nc = get_next_coord(my_head, move)
        nx, ny = nc['x'], nc['y']
        if not (0 <= nx < board_w and 0 <= ny < board_h):
            continue
        if (nx, ny) in occupied:
            continue
        if (nx, ny) not in danger_squares:
            truly_safe.append(move)

    in_risky_fallback = (len(truly_safe) == 0)

    best_move     = safe_moves[0]
    highest_score = -math.inf

    for move in safe_moves:
        elapsed = time.perf_counter() - start_time
        if elapsed > 0.200:
            print(f"WARNING: Time cutoff reached at {elapsed*1000:.1f}ms, "
                  f"aborting deep evaluation (returning '{best_move}')")
            return best_move

        score = evaluate_move(move, game_state,
                              in_risky_fallback=in_risky_fallback,
                              profiles=game_profiles)
        if DEBUG:
            print(f"  {move:5} -> {score:.2f}")
        if score > highest_score:
            highest_score = score
            best_move     = move

    if DEBUG:
        print(f"  Chosen: {best_move}\n")
    return best_move


# =================================================================
#  TAIL-VACATING HELPER
# =================================================================

def _tail_will_stay(snake, board):
    return snake['health'] == 100


# =================================================================
#  BOX-OUT DETECTOR
# =================================================================

def _detect_box_out(next_head, enemy, my_length, board_w, board_h):
    if my_length <= enemy['length']:
        return False, None

    ehx = enemy['head']['x']
    ehy = enemy['head']['y']

    on_left   = (ehx == 0)
    on_right  = (ehx == board_w - 1)
    on_bottom = (ehy == 0)
    on_top    = (ehy == board_h - 1)

    if not (on_left or on_right or on_bottom or on_top):
        return False, None

    if len(enemy['body']) < 2:
        return False, None

    b1 = enemy['body'][1]
    dx = ehx - b1['x']
    dy = ehy - b1['y']

    parallel = (
        ((on_left or on_right)  and dx == 0 and dy != 0) or
        ((on_top  or on_bottom) and dy == 0 and dx != 0)
    )
    if not parallel:
        return False, None

    enx = ehx + dx
    eny = ehy + dy

    nhx = next_head['x']
    nhy = next_head['y']

    matched = (
        (on_left   and nhx == enx + 1 and nhy == eny) or
        (on_right  and nhx == enx - 1 and nhy == eny) or
        (on_bottom and nhx == enx     and nhy == eny + 1) or
        (on_top    and nhx == enx     and nhy == eny - 1)
    )

    if not matched:
        return False, None

    return True, {'x': enx, 'y': eny}


# =================================================================
#  SAFE MOVE FILTER
# =================================================================

def get_safe_moves(game_state):
    my_snake  = game_state['you']
    my_head   = my_snake['head']
    board     = game_state['board']
    board_w   = board['width']
    board_h   = board['height']
    my_length = my_snake['length']
    all_moves = ["up", "down", "left", "right"]

    occupied = set()
    for snake in board['snakes']:
        tail_stays = _tail_will_stay(snake, board)
        for i, part in enumerate(snake['body']):
            if not tail_stays and i == len(snake['body']) - 1:
                continue
            occupied.add((part['x'], part['y']))

    danger_squares = set()
    for snake in board['snakes']:
        if snake['id'] == my_snake['id']:
            continue
        if snake['length'] >= my_length:
            for move in all_moves:
                nc = get_next_coord(snake['head'], move)
                nx, ny = nc['x'], nc['y']
                if 0 <= nx < board_w and 0 <= ny < board_h:
                    danger_squares.add((nx, ny))

    safe  = []
    risky = []

    for move in all_moves:
        nc = get_next_coord(my_head, move)
        nx, ny = nc['x'], nc['y']

        if not (0 <= nx < board_w and 0 <= ny < board_h):
            continue
        if (nx, ny) in occupied:
            continue
        if (nx, ny) in danger_squares:
            risky.append(move)
        else:
            safe.append(move)

    return safe if safe else risky


# =================================================================
#  MOVE SCORER
# =================================================================

_THREAT_PENALTY = {1: -120, 2: -70, 3: -25, 4: -8}

_BOX_OUT_BONUS = 150
_CORNER_BONUS  =  60


def evaluate_move(move, game_state, in_risky_fallback=False, profiles=None):
    score      = 0
    my_snake   = game_state['you']
    board      = game_state['board']
    next_head  = get_next_coord(my_snake['head'], move)
    health     = my_snake['health']
    my_length  = my_snake['length']
    board_size = board['width'] * board['height']

    # ------------------------------------------------------------------
    # 1. FLOOD FILL
    # ------------------------------------------------------------------
    space       = calculate_flood_fill(next_head, game_state)
    space_ratio = space / board_size

    if space == 0:
        return -9999

    if space < my_length:
        trap_severity = 1.0 - (space / my_length)
        return -500 - trap_severity * 500

    score += space_ratio * 100

    if space < 2 * my_length:
        tightness = 1.0 - (space / (2.0 * my_length))
        score -= tightness * 40

    # ------------------------------------------------------------------
    # 1b. CORRIDOR TRAP DETECTION
    #
    #  Flood fill counts total reachable space but cannot distinguish
    #  between a wide-open region and a long dead-end corridor.  A move
    #  that enters a one-way corridor (only 1 open exit from next_head)
    #  is dangerous regardless of how many tiles lie beyond it: if an
    #  enemy or our own body closes the entrance next turn, all that
    #  flood-fill space becomes unreachable and we starve.
    #
    #  Implementation:
    #    Build the same occupied set used by flood fill, then count how
    #    many of next_head's 4 neighbours are passable.  The square we
    #    came from (current head, now body[1]) is already in occupied,
    #    so it is correctly excluded from the count without special-casing.
    #
    #  Penalty scale:
    #    • 1 open exit  → corridor; penalty grows as available space shrinks.
    #      At space_ratio 0.8 (open board) the penalty is mild (~10 pts) —
    #      a long corridor on an open board is survivable.
    #      At space_ratio 0.2 (cramped board) the penalty is severe (~50 pts) —
    #      entering a one-way street when space is already tight is very likely
    #      fatal.
    #    • 0 open exits → complete dead end; flood fill should have returned
    #      space ≤ 1 and bailed earlier, but this provides belt-and-suspenders
    #      coverage with a fixed -150 override.
    #    • 2+ open exits → no penalty (multiple escape routes available).
    # ------------------------------------------------------------------
    _corr_occupied = set()
    for _cs in board['snakes']:
        _ct = _tail_will_stay(_cs, board)
        for _ci, _cp in enumerate(_cs['body']):
            if not _ct and _ci == len(_cs['body']) - 1:
                continue
            _corr_occupied.add((_cp['x'], _cp['y']))

    passable_exits = 0
    for _cdx, _cdy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        _cnx = next_head['x'] + _cdx
        _cny = next_head['y'] + _cdy
        if not (0 <= _cnx < board['width'] and 0 <= _cny < board['height']):
            continue
        if (_cnx, _cny) in _corr_occupied:
            continue
        passable_exits += 1

    if passable_exits == 0:
        # Complete dead end — belt-and-suspenders (flood fill should
        # have caught this, but guard anyway)
        score -= 150
    elif passable_exits == 1:
        # One-way corridor — penalty scales with board tightness.
        # Formula: flat base of 10 pts + up to 40 pts as space shrinks.
        corridor_penalty = 10 + 40 * (1.0 - space_ratio)
        score -= corridor_penalty

    # ------------------------------------------------------------------
    # 2. FOOD
    #
    # FIX 1: competition zeroing uses > instead of >=
    #   The original used `snake['length'] >= my_length` to set
    #   bigger_enemy_can_contest=True. Because all snakes start at the
    #   same length, this ALWAYS fired early game, setting competition_
    #   factor=0.0 on every contested food tile. Thunga effectively
    #   stopped pursuing food after turn 2 since no food ever scored > 0.
    #   Fix: only truly LARGER snakes (strictly >) should zero out food.
    #
    # FIX 2: food_urgency floor raised from 0.40 → 0.60
    #   At full health the old formula gave 0.40, meaning food at
    #   distance 3 scored only ~8 pts — less than the center control
    #   bonus. Food was systematically undervalued versus space-keeping,
    #   causing Thunga to ignore it when healthy. Raising the floor to
    #   0.60 ensures food is always a meaningful pull signal.
    # ------------------------------------------------------------------
    base         = (100 - health) / 100
    food_urgency = 0.60 + 0.40 * (base ** 1.0)    # FIX 2: was 0.40 + 0.60 * base

    best_food_score = 0
    if board['food']:
        board_w = board['width']
        board_h = board['height']
        next_is_danger = False
        for snake in board['snakes']:
            if snake['id'] == my_snake['id']:
                continue
            if snake['length'] >= my_length:
                for mv2 in ["up", "down", "left", "right"]:
                    nc2 = get_next_coord(snake['head'], mv2)
                    if nc2['x'] == next_head['x'] and nc2['y'] == next_head['y']:
                        next_is_danger = True
                        break

        for food in board['food']:
            path_len = astar_distance(next_head, food, game_state)
            if path_len is None:
                continue

            my_steps = path_len + 1
            enemy_min_dist = 999
            bigger_enemy_can_contest = False
            for s in board['snakes']:
                if s['id'] == my_snake['id']:
                    continue
                ed = manhattan(s['head'], food)
                if ed < enemy_min_dist:
                    enemy_min_dist = ed
                # FIX 1: was `s['length'] >= my_length`
                # Equal-length snakes are competitors, not dominators —
                # we still try for the food (reduced factor), only truly
                # larger snakes warrant a full zero-out.
                if s['length'] > my_length and ed <= my_steps:
                    bigger_enemy_can_contest = True

            if bigger_enemy_can_contest:
                competition_factor = 0.0
            elif enemy_min_dist <= my_steps:
                competition_factor = 0.35
            else:
                competition_factor = 1.0

            food_at_next = (food['x'] == next_head['x'] and food['y'] == next_head['y'])
            if in_risky_fallback and food_at_next and next_is_danger:
                competition_factor = 0.0

            candidate = (1 / (path_len + 1)) * food_urgency * 80 * competition_factor
            if candidate > best_food_score:
                best_food_score = candidate

    score += best_food_score

    if health <= 25 and best_food_score > 0:
        score += best_food_score * 0.5

    # ------------------------------------------------------------------
    # 3. CENTER CONTROL
    # ------------------------------------------------------------------
    cx, cy       = board['width'] // 2, board['height'] // 2
    center_dist  = abs(next_head['x'] - cx) + abs(next_head['y'] - cy)
    max_c_dist   = cx + cy
    center_weight = 20 * max(0.3, 1.0 - food_urgency * 0.7)
    score        += (1 - center_dist / max_c_dist) * center_weight

    # ------------------------------------------------------------------
    # 3b. WALL PROXIMITY PENALTY
    # ------------------------------------------------------------------
    wall_penalty_weight = max(0.3, 1.0 - food_urgency * 0.7)
    if next_head['x'] == 0 or next_head['x'] == board['width'] - 1:
        score -= 25 * wall_penalty_weight
    if next_head['y'] == 0 or next_head['y'] == board['height'] - 1:
        score -= 25 * wall_penalty_weight

    # ------------------------------------------------------------------
    # 4. ENEMY THREAT / OPPORTUNITY
    # ------------------------------------------------------------------
    for snake in board['snakes']:
        if snake['id'] == my_snake['id']:
            continue

        dist = manhattan(next_head, snake['head'])

        if snake['length'] >= my_length:
            penalty = _THREAT_PENALTY.get(dist, 0)
            score  += penalty
        else:
            if dist <= 2 and space_ratio > 0.35:
                score += 25

    # ------------------------------------------------------------------
    # 4b. DOUBLE-CUT DETECTION
    #
    #  Section 4 checks each enemy independently and assigns graduated
    #  penalties per snake. This is blind to combined threats: two enemies
    #  each sitting 3-4 tiles away score small individual penalties, but
    #  together they can cover all 4 of next_head's neighbours in one turn,
    #  leaving no legal escape. This is the "double-cut" — death by
    #  coordinated enclosure even when no single enemy looks dangerous.
    #
    #  Algorithm (O(snakes)):
    #    Build a local danger set from all equal/larger enemy heads, then
    #    count how many of next_head's 4 neighbours fall inside it.
    #    Out-of-bounds squares count as covered — a wall removes an escape
    #    route just as surely as an enemy head zone. This means a snake
    #    pressed against a wall only needs 2 enemies to complete a full
    #    double-cut, which is accurate: the wall is one side of the box.
    #
    #  Penalty tiers (additive on top of section 4 individual penalties):
    #    3 neighbours covered → severe (-80).  One exit remains but the
    #      enclosure is already half-complete; that exit likely closes
    #      next turn too.
    #    4 neighbours covered → critical (-160).  Every exit is a
    #      head-to-head with an equal/larger snake — near-certain death.
    #
    #  These penalties are intentionally large enough to override food
    #  urgency and center bonuses, since no positional advantage is worth
    #  walking into a box with no exit.
    # ------------------------------------------------------------------
    _dc_danger = set()
    for _dc_snake in board['snakes']:
        if _dc_snake['id'] == my_snake['id']:
            continue
        if _dc_snake['length'] >= my_length:
            for _dc_move in ["up", "down", "left", "right"]:
                _dc_nc = get_next_coord(_dc_snake['head'], _dc_move)
                _dc_nx, _dc_ny = _dc_nc['x'], _dc_nc['y']
                if 0 <= _dc_nx < board['width'] and 0 <= _dc_ny < board['height']:
                    _dc_danger.add((_dc_nx, _dc_ny))

    covered_neighbours = 0
    for _dc_dx, _dc_dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        _dc_nx = next_head['x'] + _dc_dx
        _dc_ny = next_head['y'] + _dc_dy
        # Out-of-bounds counts as covered — walls block escape routes
        if not (0 <= _dc_nx < board['width'] and 0 <= _dc_ny < board['height']):
            covered_neighbours += 1
        elif (_dc_nx, _dc_ny) in _dc_danger:
            covered_neighbours += 1

    if covered_neighbours >= 4:
        score -= 160   # every exit threatened — near-certain death
    elif covered_neighbours == 3:
        score -= 80    # one exit left; enclosure closing fast

    # ------------------------------------------------------------------
    # 5. VORONOI TERRITORY
    # ------------------------------------------------------------------
    voronoi       = calculate_voronoi_space(next_head, game_state)
    voronoi_ratio = voronoi / board_size
    score        += voronoi_ratio * 30

    # ------------------------------------------------------------------
    # 5b. BOX-OUT OVERLAY
    # ------------------------------------------------------------------
    for snake in board['snakes']:
        if snake['id'] == my_snake['id']:
            continue

        box_out, _proj = _detect_box_out(
            next_head, snake, my_length,
            board['width'], board['height']
        )

        if not box_out:
            continue

        if space_ratio <= 0.30:
            continue

        dist_to_enemy = manhattan(next_head, snake['head'])
        if dist_to_enemy <= 2 and space_ratio > 0.35:
            score -= 25

        score += _BOX_OUT_BONUS

        ex, ey = snake['head']['x'], snake['head']['y']
        bw, bh = board['width'], board['height']
        corners = [(0, 0), (0, bh - 1), (bw - 1, 0), (bw - 1, bh - 1)]
        min_corner_dist = min(
            abs(ex - crx) + abs(ey - cry) for crx, cry in corners
        )
        if min_corner_dist <= 2:
            score += _CORNER_BONUS

    # ------------------------------------------------------------------
    # 6. TACTICAL OVERLAY (Profiler-Driven)
    # ------------------------------------------------------------------
    if profiles:
        for snake in board['snakes']:
            if snake['id'] == my_snake['id']:
                continue

            sid = snake['id']
            p   = profiles.get(sid)

            if p is None or p['archetype'] is None:
                continue

            archetype  = p['archetype']
            confidence = p['confidence']
            enemy_dist = manhattan(next_head, snake['head'])

            if enemy_dist >= _INTERACTION_RADIUS:
                continue

            if archetype == 'vacuum':
                for food in board['food']:
                    vac_dist = manhattan(snake['head'], food)
                    my_dist  = manhattan(next_head,    food)
                    if vac_dist <= 2 and vac_dist < my_dist and my_dist == 1:
                        score += 40 * confidence

            elif archetype == 'camper':
                camper_hx = snake['head']['x']
                camper_hy = snake['head']['y']
                bw = board['width']
                bh = board['height']

                camper_on_wall = (
                    camper_hx == 0 or camper_hx == bw - 1 or
                    camper_hy == 0 or camper_hy == bh - 1
                )

                if camper_on_wall:
                    if next_head['x'] == 0 or next_head['x'] == bw - 1:
                        score += 25 * wall_penalty_weight
                    if next_head['y'] == 0 or next_head['y'] == bh - 1:
                        score += 25 * wall_penalty_weight

                    inner_x = camper_hx
                    inner_y = camper_hy
                    if camper_hx == 0:        inner_x = 1
                    elif camper_hx == bw - 1: inner_x = bw - 2
                    if camper_hy == 0:        inner_y = 1
                    elif camper_hy == bh - 1: inner_y = bh - 2

                    if next_head['x'] == inner_x and next_head['y'] == inner_y:
                        score += 30 * confidence

            elif archetype == 'aggressor':
                if snake['length'] >= my_length:
                    extra = _THREAT_PENALTY.get(enemy_dist, 0) * 2
                    score += extra

                if (space_ratio < 0.5 and
                        len(my_snake['body']) > 1 and
                        not _tail_will_stay(my_snake, board)):
                    my_tail   = my_snake['body'][-1]
                    tail_dist = (abs(next_head['x'] - my_tail['x']) +
                                 abs(next_head['y'] - my_tail['y']))
                    if tail_dist <= 3:
                        score += 35 * confidence

            elif archetype == 'area_control':
                # Keep in sync with the new floor (0.60 + 0.40 * base)
                adjusted_health = max(0, health - 25)
                adj_base        = (100 - adjusted_health) / 100
                adj_urgency     = 0.60 + 0.40 * adj_base
                urgency_delta   = adj_urgency - food_urgency
                if best_food_score > 0:
                    score += best_food_score * urgency_delta * confidence

            elif archetype == 'erratic':
                if snake['length'] >= my_length and 2 <= enemy_dist <= 4:
                    cancelled = _THREAT_PENALTY.get(enemy_dist, 0)
                    score    -= cancelled

    return score


# =================================================================
#  FLOOD FILL
# =================================================================

def calculate_flood_fill(head, game_state):
    board  = game_state['board']
    width  = board['width']
    height = board['height']

    occupied = set()
    for snake in game_state['board']['snakes']:
        tail_stays = _tail_will_stay(snake, board)
        for i, part in enumerate(snake['body']):
            if not tail_stays and i == len(snake['body']) - 1:
                continue
            occupied.add((part['x'], part['y']))

    visited = set()
    queue   = deque([(head['x'], head['y'])])

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited:
            continue
        if not (0 <= x < width and 0 <= y < height):
            continue
        if (x, y) in occupied:
            continue
        visited.add((x, y))
        queue.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])

    return len(visited)


# =================================================================
#  VORONOI SPACE
# =================================================================

def calculate_voronoi_space(head, game_state):
    board   = game_state['board']
    width   = board['width']
    height  = board['height']
    my_id   = game_state['you']['id']

    occupied = set()
    for snake in board['snakes']:
        tail_stays = _tail_will_stay(snake, board)
        for i, part in enumerate(snake['body']):
            if not tail_stays and i == len(snake['body']) - 1:
                continue
            occupied.add((part['x'], part['y']))

    dist_map = {}
    heap     = []

    our_pos = (head['x'], head['y'])
    if our_pos not in occupied:
        dist_map[our_pos] = (0, 'our')
        heapq.heappush(heap, (0, our_pos, 'our'))

    for snake in board['snakes']:
        if snake['id'] == my_id:
            continue
        epos = (snake['head']['x'], snake['head']['y'])
        if epos in occupied:
            continue
        if epos not in dist_map:
            dist_map[epos] = (0, snake['id'])
            heapq.heappush(heap, (0, epos, snake['id']))
        else:
            existing_d, existing_owner = dist_map[epos]
            if existing_d == 0 and existing_owner != snake['id']:
                dist_map[epos] = (0, 'tie')

    while heap:
        d, pos, owner = heapq.heappop(heap)

        existing_d, existing_owner = dist_map.get(pos, (None, None))
        if existing_d is None or d > existing_d:
            continue
        if existing_owner == 'tie' and owner != 'tie':
            continue

        x, y = pos
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in occupied:
                continue
            new_d   = d + 1
            new_pos = (nx, ny)
            if new_pos not in dist_map:
                dist_map[new_pos] = (new_d, owner)
                heapq.heappush(heap, (new_d, new_pos, owner))
            else:
                prev_d, prev_owner = dist_map[new_pos]
                if new_d < prev_d:
                    dist_map[new_pos] = (new_d, owner)
                    heapq.heappush(heap, (new_d, new_pos, owner))
                elif new_d == prev_d and prev_owner != owner and prev_owner != 'tie':
                    dist_map[new_pos] = (new_d, 'tie')

    return sum(1 for (_, owner) in dist_map.values() if owner == 'our')


# =================================================================
#  A* PATHFINDING
# =================================================================

def astar_distance(start, goal, game_state):
    board  = game_state['board']
    width  = board['width']
    height = board['height']

    occupied = set()
    for snake in game_state['board']['snakes']:
        tail_stays = _tail_will_stay(snake, board)
        for i, part in enumerate(snake['body']):
            if not tail_stays and i == len(snake['body']) - 1:
                continue
            occupied.add((part['x'], part['y']))

    start_pos = (start['x'], start['y'])
    goal_pos  = (goal['x'],  goal['y'])

    h0       = manhattan(start, goal)
    heap     = [(h0, 0, start_pos)]
    g_scores = {start_pos: 0}

    while heap:
        f, g, pos = heapq.heappop(heap)

        if pos == goal_pos:
            return g

        if g > g_scores.get(pos, float('inf')):
            continue

        x, y = pos
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in occupied:
                continue
            new_g = g + 1
            if new_g < g_scores.get((nx, ny), float('inf')):
                g_scores[(nx, ny)] = new_g
                new_f = new_g + manhattan({"x": nx, "y": ny}, goal)
                heapq.heappush(heap, (new_f, new_g, (nx, ny)))

    return None


# =================================================================
#  HELPERS
# =================================================================

def get_next_coord(head, move):
    if move == "up":    return {"x": head["x"],     "y": head["y"] + 1}
    if move == "down":  return {"x": head["x"],     "y": head["y"] - 1}
    if move == "left":  return {"x": head["x"] - 1, "y": head["y"]}
    if move == "right": return {"x": head["x"] + 1, "y": head["y"]}

def manhattan(a, b):
    return abs(a['x'] - b['x']) + abs(a['y'] - b['y'])

def get_closest_food_distance(head, food_list):
    if not food_list:
        return 0
    return min(manhattan(head, food) for food in food_list)
