"""A clear, paced pygame board for playing against a trained checkpoint."""

from __future__ import annotations

import numpy as np
import torch

from .constants import BAR, BLACK, BOARD_POINTS, PASS_ACTION, WHITE, action_index, decode_action
from .env import BackgammonEnv
from .game import destination_for
from .inference import choose_action, value_of_observation
from .model import load_checkpoint, resolve_device


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 840
BOARD_RECT = (35, 100, 900, 640)
POINT_LEFT = 75
POINT_RIGHT = 895
POINT_TOP = 125
POINT_BOTTOM = 715
CENTER_Y = (POINT_TOP + POINT_BOTTOM) // 2
# There are six points, a full-width bar gap, and six points.
CELL_WIDTH = (POINT_RIGHT - POINT_LEFT) / 13.0
BAR_SLOT = 6
SIDEBAR_RECT = (965, 30, 280, 780)
AI_MOVE_DELAY_MS = 950
EVALUATION_REFRESH_MS = 180

BACKGROUND = (24, 29, 38)
PANEL = (32, 39, 51)
BOARD = (171, 103, 53)
BOARD_EDGE = (239, 190, 112)
LIGHT_POINT = (235, 185, 108)
DARK_POINT = (113, 56, 43)
WHITE_CHECKER = (244, 239, 219)
WHITE_EDGE = (70, 70, 73)
BLACK_CHECKER = (34, 39, 49)
BLACK_EDGE = (213, 177, 101)
TEXT = (240, 244, 250)
MUTED = (167, 180, 198)
ACCENT = (76, 190, 166)
HIGHLIGHT = (255, 218, 92)
POSITIVE = (84, 205, 145)
NEGATIVE = (226, 105, 105)
DIE_RECT_SIZE = (42, 38)
DIE_ORIGIN = (985, 210)


def _point_slot(point: int) -> float:
    """Map an internal point to the standard physical board layout.

    Internal point 0 is displayed as point 1 at the bottom-right.  Internal
    point 11 is displayed as point 12 at the bottom-left.  Points 13..24 run
    left-to-right along the top row.  This is the usual backgammon board view,
    rather than two rows numbered in the same direction.
    """
    local = point - 12 if point >= 12 else 11 - point
    return float(local if local < 6 else local + 1)


def point_center(point: int) -> int:
    return int(POINT_LEFT + (_point_slot(point) + 0.5) * CELL_WIDTH)


def bar_center() -> int:
    return int(POINT_LEFT + (BAR_SLOT + 0.5) * CELL_WIDTH)


def source_at_position(x: int, y: int) -> int | None:
    if y < POINT_TOP - 22 or y > POINT_BOTTOM + 22:
        return None
    for point in range(BOARD_POINTS):
        center = point_center(point)
        if abs(x - center) <= CELL_WIDTH * 0.45:
            if point < 12 and y >= CENTER_Y:
                return point
            if point >= 12 and y <= CENTER_Y:
                return point
    if abs(x - bar_center()) <= CELL_WIDTH * 0.48:
        return BAR
    return None


def _text(surface, font, value: str, position: tuple[int, int], color=TEXT) -> None:
    surface.blit(font.render(value, True, color), position)


def _center_text(surface, font, value: str, center: tuple[int, int], color=TEXT) -> None:
    image = font.render(value, True, color)
    surface.blit(image, image.get_rect(center=center))


def _legal_dice_for_source(mask: np.ndarray, source: int) -> list[int]:
    dice: list[int] = []
    for raw_action in np.flatnonzero(mask):
        decoded = decode_action(int(raw_action))
        if decoded is not None and decoded[0] == source:
            dice.append(decoded[1])
    return sorted(set(dice))


def _die_rects() -> list[object]:
    """Return the four clickable dice rectangles without importing pygame."""
    # Rect construction is done by the caller's pygame module.
    return [
        (DIE_ORIGIN[0] + (index % 4) * 49, DIE_ORIGIN[1], DIE_RECT_SIZE[0], DIE_RECT_SIZE[1])
        for index in range(4)
    ]


def _draw_checker(surface, pygame, center: tuple[int, int], white: bool, radius: int = 23) -> None:
    fill = WHITE_CHECKER if white else BLACK_CHECKER
    edge = WHITE_EDGE if white else BLACK_EDGE
    pygame.draw.circle(surface, edge, center, radius + 2)
    pygame.draw.circle(surface, fill, center, radius)
    pygame.draw.circle(surface, edge, center, radius, width=1)


def _draw_board(surface, pygame, env: BackgammonEnv, fonts, selected_source: int | None) -> None:
    small, normal, large = fonts
    board_left, board_top, board_width, board_height = BOARD_RECT
    board_bottom = board_top + board_height
    pygame.draw.rect(surface, BOARD, BOARD_RECT, border_radius=12)
    pygame.draw.rect(surface, BOARD_EDGE, BOARD_RECT, width=3, border_radius=12)

    for point in range(BOARD_POINTS):
        x = point_center(point)
        local = point - 12 if point >= 12 else 11 - point
        point_color = LIGHT_POINT if local % 2 == 0 else DARK_POINT
        if point < 12:
            polygon = [
                (x - int(CELL_WIDTH * 0.46), CENTER_Y - 7),
                (x + int(CELL_WIDTH * 0.46), CENTER_Y - 7),
                (x, board_bottom - 16),
            ]
            label_y = board_bottom - 13
        else:
            polygon = [
                (x - int(CELL_WIDTH * 0.46), CENTER_Y + 7),
                (x + int(CELL_WIDTH * 0.46), CENTER_Y + 7),
                (x, board_top + 16),
            ]
            label_y = board_top + 13
        pygame.draw.polygon(surface, point_color, polygon)
        _center_text(surface, small, str(point + 1), (x, label_y), (252, 231, 177))

    # A full-width gap between points 18/19 and 6/7, as on a standard board.
    x_bar = bar_center()
    pygame.draw.rect(
        surface,
        (121, 68, 43),
        (x_bar - int(CELL_WIDTH * 0.46), board_top + 8, int(CELL_WIDTH * 0.92), board_height - 16),
        border_radius=8,
    )
    _center_text(surface, small, "BAR", (x_bar, CENTER_Y), (246, 216, 146))

    mask = env.action_mask() if not env.done else np.zeros(env.action_size, dtype=np.bool_)
    legal_sources = {
        decoded[0]
        for raw_action in np.flatnonzero(mask)
        if (decoded := decode_action(int(raw_action))) is not None
    }
    for source in legal_sources:
        if source == BAR:
            pygame.draw.rect(
                surface,
                HIGHLIGHT,
                (x_bar - int(CELL_WIDTH * 0.49), board_top + 5, int(CELL_WIDTH * 0.98), board_height - 10),
                width=3,
                border_radius=8,
            )
        else:
            x = point_center(source)
            y = board_bottom - 46 if source < 12 else board_top + 46
            pygame.draw.circle(surface, HIGHLIGHT, (x, y), 28, width=3)
    if selected_source is not None:
        x = x_bar if selected_source == BAR else point_center(selected_source)
        y = CENTER_Y if selected_source == BAR else (board_bottom - 46 if selected_source < 12 else board_top + 46)
        pygame.draw.circle(surface, ACCENT, (x, y), 32, width=4)

    # Checkers on the points. Five are stacked visually; the number label
    # makes larger stacks unambiguous.
    for point, value in enumerate(env.state.board):
        count = abs(value)
        if count == 0:
            continue
        white = value > 0
        x = point_center(point)
        visible = min(count, 5)
        for stack_index in range(visible):
            if point < 12:
                y = board_bottom - 60 - stack_index * 35
            else:
                y = board_top + 60 + stack_index * 35
            _draw_checker(surface, pygame, (x, y), white)
        if count > 5:
            y = board_bottom - 60 if point < 12 else board_top + 60
            _center_text(surface, small, str(count), (x, y), BLACK_CHECKER if white else WHITE_CHECKER)

    # Bar checkers are separated by side of the bar so both colours remain
    # visible when several checkers are hit.
    for player, count in ((WHITE, env.state.bar[WHITE]), (BLACK, env.state.bar[BLACK])):
        for index in range(min(count, 4)):
            y = CENTER_Y - 58 - index * 35 if player == WHITE else CENTER_Y + 58 + index * 35
            _draw_checker(surface, pygame, (x_bar, y), player == WHITE, radius=19)
        if count:
            label_y = CENTER_Y - 57 if player == WHITE else CENTER_Y + 57
            _center_text(surface, small, str(count), (x_bar + 30, label_y), TEXT)


def _turn_dice_with_usage(env: BackgammonEnv) -> list[tuple[int, bool]]:
    """Return (die, used) for each die in the original roll."""
    remaining = list(env.state.dice)
    result: list[tuple[int, bool]] = []
    for die in env.state.turn_dice:
        if die in remaining:
            remaining.remove(die)
            result.append((die, False))
        else:
            result.append((die, True))
    return result


def _draw_dice(surface, pygame, fonts, env: BackgammonEnv) -> list[object]:
    small, normal, _ = fonts
    rects = []
    for index, (die, used) in enumerate(_turn_dice_with_usage(env)):
        x, y, width, height = _die_rects()[index]
        rect = pygame.Rect(x, y, width, height)
        rects.append(rect)
        fill = (92, 101, 115) if used else (239, 241, 231)
        text_color = (190, 197, 207) if used else (35, 43, 54)
        pygame.draw.rect(surface, fill, rect, border_radius=7)
        _center_text(surface, normal, str(die), rect.center, text_color)
        if used:
            pygame.draw.line(surface, NEGATIVE, (rect.left + 7, rect.bottom - 7), (rect.right - 7, rect.top + 7), width=2)
    return rects


def _draw_evaluation(surface, pygame, fonts, evaluation: float) -> None:
    small, normal, _ = fonts
    x, y, width, height = 985, 310, 240, 25
    value = float(np.clip(evaluation, -1.0, 1.0))
    ratio = (value + 1.0) / 2.0
    pygame.draw.rect(surface, (55, 65, 80), (x, y, width, height), border_radius=6)
    fill_color = POSITIVE if value >= 0 else NEGATIVE
    fill_width = max(2, int(width * ratio))
    pygame.draw.rect(surface, fill_color, (x, y, fill_width, height), border_radius=6)
    pygame.draw.line(surface, (238, 242, 247), (x + width // 2, y - 3), (x + width // 2, y + height + 3), width=2)
    _text(surface, small, "AI evaluation", (x, y - 28), MUTED)
    _text(surface, small, "loss", (x, y + 31), MUTED)
    _center_text(surface, normal, f"{value:+.2f}", (x + width // 2, y + height + 14), fill_color)
    _text(surface, small, "win", (x + width - 26, y + 31), MUTED)


def _draw_sidebar(
    surface,
    pygame,
    env: BackgammonEnv,
    fonts,
    human_player: int,
    message: str,
    pending_source: int | None,
    evaluation: float,
    last_ai_move: str,
    restart_rect,
    pass_rect,
) -> list[object]:
    small, normal, large = fonts
    pygame.draw.rect(surface, PANEL, SIDEBAR_RECT, border_radius=14)
    pygame.draw.rect(surface, (61, 76, 95), SIDEBAR_RECT, width=2, border_radius=14)
    _text(surface, large, "BG • RL", (990, 55), ACCENT)
    _text(surface, small, "BACKGAMMON  •  HUMAN VS AI", (991, 91), MUTED)

    current_name = "WHITE" if env.current_player == WHITE else "BLACK"
    current_color = WHITE_CHECKER if env.current_player == WHITE else (226, 139, 91)
    _text(surface, normal, f"Turn: {current_name}", (990, 125), current_color)
    _text(surface, small, "You" if env.current_player == human_player else "Neural agent", (990, 153), ACCENT if env.current_player == human_player else HIGHLIGHT)
    _text(surface, small, "White direction: 1 → 24", (990, 177), MUTED)
    _text(surface, small, "Black direction: 24 → 1", (990, 195), MUTED)

    _text(surface, small, "Rolled dice  (click a die after selecting)", (990, 250), MUTED)
    die_rects = _draw_dice(surface, pygame, fonts, env)
    remaining_text = "Remaining: " + (" ".join(map(str, env.state.dice)) if env.state.dice else "none")
    _text(surface, small, remaining_text, (990, 270), TEXT)

    _draw_evaluation(surface, pygame, fonts, evaluation)
    _text(surface, small, "Network value from the AI's point of view", (990, 375), MUTED)
    _text(surface, small, "not a guaranteed probability", (990, 392), MUTED)

    _text(surface, small, "Borne off", (990, 420), MUTED)
    _text(surface, normal, f"White: {env.state.off[WHITE]:2d}", (990, 447), WHITE_CHECKER)
    _text(surface, normal, f"Black: {env.state.off[BLACK]:2d}", (990, 476), (226, 139, 91))
    _text(surface, small, f"Bar: W {env.state.bar[WHITE]}  B {env.state.bar[BLACK]}", (990, 505), MUTED)

    _text(surface, small, "Last AI move", (990, 540), MUTED)
    move_rect = pygame.Rect(985, 563, 240, 47)
    pygame.draw.rect(surface, (42, 51, 66), move_rect, border_radius=8)
    move_words = last_ai_move.split()
    move_lines: list[str] = []
    line = ""
    for word in move_words:
        candidate = f"{line} {word}".strip()
        if small.size(candidate)[0] > 225 and line:
            move_lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        move_lines.append(line)
    for index, line in enumerate(move_lines[:2]):
        _text(surface, small, line, (993, 572 + index * 17), TEXT)

    message_rect = pygame.Rect(985, 625, 240, 55)
    pygame.draw.rect(surface, (42, 51, 66), message_rect, border_radius=8)
    words = message.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if small.size(candidate)[0] > 225 and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for index, line in enumerate(lines[:2]):
        _text(surface, small, line, (993, 634 + index * 18), TEXT)

    pygame.draw.rect(surface, (55, 68, 84), restart_rect, border_radius=7)
    _center_text(surface, small, "R  New game", restart_rect.center, TEXT)
    pygame.draw.rect(surface, (55, 68, 84), pass_rect, border_radius=7)
    _center_text(surface, small, "SPACE  Pass", pass_rect.center, TEXT)

    if pending_source is not None:
        source_text = "BAR" if pending_source == BAR else f"point {pending_source + 1}"
        _text(surface, small, f"Selected {source_text} • choose a die", (990, 780), HIGHLIGHT)
    return die_rects


def _evaluate_for_ai(model, env: BackgammonEnv, device: torch.device, ai_player: int) -> float:
    if env.done:
        return 1.0 if env.winner == ai_player else -1.0
    raw_value = value_of_observation(model, env.observation(), device)
    # The observation/value is always from the player whose turn it is. Flip
    # it when the human is to move so the bar always means AI perspective.
    return raw_value if env.current_player == ai_player else -raw_value


def _move_description(player: int, source: int, die: int) -> str:
    source_text = "BAR" if source == BAR else f"P{source + 1}"
    destination = destination_for(source, die, player)
    destination_text = "OFF" if destination is None else f"P{destination + 1}"
    return f"{'W' if player == WHITE else 'B'}: {source_text} → {destination_text} (die {die})"


def run_game(
    checkpoint_path: str,
    human_player: int = WHITE,
    device_name: str = "auto",
    seed: int = 123,
    ai_delay_ms: int = AI_MOVE_DELAY_MS,
    search_samples: int = 2,
) -> None:
    """Open the local GUI and run a paced human-vs-checkpoint game."""
    try:
        import pygame
    except ImportError as error:  # pragma: no cover - depends on local extras
        raise RuntimeError("pygame is not installed; run: python -m pip install pygame") from error

    if ai_delay_ms < 0:
        raise ValueError("ai_delay_ms must be non-negative")
    if search_samples < 0:
        raise ValueError("search_samples must be non-negative")
    device = resolve_device(device_name)
    model, _ = load_checkpoint(checkpoint_path, device=device)
    env = BackgammonEnv(seed=seed)
    env.reset(starting_player=human_player)

    pygame.init()
    pygame.display.set_caption("Backgammon RL — Human vs AI")
    surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    small = pygame.font.SysFont("segoeui", 16)
    normal = pygame.font.SysFont("segoeui", 20)
    large = pygame.font.SysFont("segoeui", 28, bold=True)
    fonts = (small, normal, large)

    restart_rect = pygame.Rect(985, 700, 240, 36)
    pass_rect = pygame.Rect(985, 742, 240, 36)
    pending_source: int | None = None
    message = "Click a highlighted checker to move."
    last_ai_move = "Waiting for the first AI move"
    evaluation = _evaluate_for_ai(model, env, device, 1 - human_player)
    ai_player = 1 - human_player
    running = True
    last_ai_time = pygame.time.get_ticks()
    last_evaluation_time = 0

    def restart() -> None:
        nonlocal pending_source, message, last_ai_move, last_ai_time, last_evaluation_time
        env.reset(starting_player=human_player)
        pending_source = None
        message = "Click a highlighted checker to move."
        last_ai_move = "Waiting for the first AI move"
        last_ai_time = pygame.time.get_ticks()
        last_evaluation_time = 0

    def apply_action(action: int, actor_is_ai: bool = False) -> None:
        nonlocal pending_source, message, last_ai_time, last_ai_move
        player = env.current_player
        decoded = decode_action(action)
        if actor_is_ai and decoded is not None:
            last_ai_move = _move_description(player, decoded[0], decoded[1])
        try:
            _, _, done, info = env.step(action)
        except ValueError:
            message = "That move is not legal. Choose a highlighted option."
            pending_source = None
            return
        pending_source = None
        last_ai_time = pygame.time.get_ticks()
        if done:
            winner = "White" if info["winner"] == WHITE else "Black"
            message = f"{winner} wins! Press R for a new game."
        elif env.current_player == human_player:
            message = "Your turn. Click a highlighted checker."
        else:
            message = "AI turn — watch the dice and last-move panel."

    while running:
        die_rects = _die_rects()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    restart()
                elif event.key == pygame.K_SPACE and not env.done and env.current_player == human_player:
                    if env.action_mask()[PASS_ACTION]:
                        apply_action(PASS_ACTION)
                    else:
                        message = "A move is available; passing is not legal."
                elif (
                    event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6)
                    and pending_source is not None
                    and not env.done
                    and env.current_player == human_player
                ):
                    die = event.key - pygame.K_0
                    action = action_index(pending_source, die)
                    if env.action_mask()[action]:
                        apply_action(action)
                    else:
                        message = "That die cannot move the selected checker."
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                if restart_rect.collidepoint(x, y):
                    restart()
                elif pass_rect.collidepoint(x, y) and not env.done and env.current_player == human_player:
                    if env.action_mask()[PASS_ACTION]:
                        apply_action(PASS_ACTION)
                    else:
                        message = "A move is available; passing is not legal."
                elif not env.done and env.current_player == human_player:
                    # The dice can be clicked after a source is selected.
                    clicked_die = None
                    for index, rect_data in enumerate(die_rects[: len(env.state.turn_dice)]):
                        rect = pygame.Rect(*rect_data)
                        if rect.collidepoint(x, y):
                            clicked_die = env.state.turn_dice[index]
                            break
                    if clicked_die is not None and pending_source is not None:
                        action = action_index(pending_source, clicked_die)
                        if env.action_mask()[action]:
                            apply_action(action)
                        else:
                            message = "That die cannot move the selected checker."
                    else:
                        source = source_at_position(x, y)
                        if source is not None:
                            dice = _legal_dice_for_source(env.action_mask(), source)
                            if len(dice) == 1:
                                apply_action(action_index(source, dice[0]))
                            elif len(dice) > 1:
                                pending_source = source
                                message = "Source selected — click a die or press its number."
                            else:
                                message = "That checker cannot move with the remaining dice."

        if not env.done and env.current_player == ai_player:
            now = pygame.time.get_ticks()
            if now - last_ai_time >= ai_delay_ms:
                action = choose_action(
                    model,
                    env,
                    device,
                    deterministic=True,
                    search_samples=search_samples,
                )
                apply_action(action, actor_is_ai=True)

        now = pygame.time.get_ticks()
        if now - last_evaluation_time >= EVALUATION_REFRESH_MS:
            evaluation = _evaluate_for_ai(model, env, device, ai_player)
            last_evaluation_time = now

        surface.fill(BACKGROUND)
        _draw_board(surface, pygame, env, fonts, pending_source)
        _draw_sidebar(
            surface,
            pygame,
            env,
            fonts,
            human_player,
            message,
            pending_source,
            evaluation,
            last_ai_move,
            restart_rect,
            pass_rect,
        )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
