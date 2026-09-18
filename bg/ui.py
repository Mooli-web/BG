"""A simple, human-friendly pygame board for playing against a checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .constants import BAR, BLACK, BOARD_POINTS, PASS_ACTION, WHITE, decode_action
from .env import BackgammonEnv
from .model import load_checkpoint, resolve_device


WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
BOARD_RECT = (40, 90, 850, 620)
POINT_LEFT = 70
POINT_RIGHT = 855
POINT_TOP = 110
POINT_BOTTOM = 690
CENTER_Y = (POINT_TOP + POINT_BOTTOM) // 2
CELL_WIDTH = (POINT_RIGHT - POINT_LEFT) / 12.5
BAR_SLOT = 6.25

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
DANGER = (232, 105, 105)


def _point_slot(point: int) -> float:
    local = point if point < 12 else point - 12
    return float(local if local < 6 else local + 0.5)


def point_center(point: int) -> int:
    return int(POINT_LEFT + (_point_slot(point) + 0.5) * CELL_WIDTH)


def source_at_position(x: int, y: int) -> int | None:
    if y < POINT_TOP - 15 or y > POINT_BOTTOM + 15:
        return None
    for point in range(BOARD_POINTS):
        center = point_center(point)
        if abs(x - center) <= CELL_WIDTH * 0.46:
            if point < 12 and y >= CENTER_Y:
                return point
            if point >= 12 and y <= CENTER_Y:
                return point
    # The bar is a wider target in the middle of the board.
    bar_center = int(POINT_LEFT + (BAR_SLOT + 0.5) * CELL_WIDTH)
    if abs(x - bar_center) <= CELL_WIDTH * 0.45:
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


def _draw_checker(surface, pygame, center: tuple[int, int], white: bool, radius: int = 23) -> None:
    fill = WHITE_CHECKER if white else BLACK_CHECKER
    edge = WHITE_EDGE if white else BLACK_EDGE
    pygame.draw.circle(surface, edge, center, radius + 2)
    pygame.draw.circle(surface, fill, center, radius)
    pygame.draw.circle(surface, edge, center, radius, width=1)


def _draw_board(surface, pygame, env: BackgammonEnv, fonts, selected_source: int | None) -> None:
    small, normal, large = fonts
    pygame.draw.rect(surface, BOARD, BOARD_RECT, border_radius=12)
    pygame.draw.rect(surface, BOARD_EDGE, BOARD_RECT, width=3, border_radius=12)

    board_left, board_top, board_width, board_height = BOARD_RECT
    board_bottom = board_top + board_height
    for point in range(BOARD_POINTS):
        x = point_center(point)
        if point < 12:
            polygon = [
                (x - int(CELL_WIDTH * 0.46), CENTER_Y - 7),
                (x + int(CELL_WIDTH * 0.46), CENTER_Y - 7),
                (x, board_bottom - 14),
            ]
        else:
            polygon = [
                (x - int(CELL_WIDTH * 0.46), CENTER_Y + 7),
                (x + int(CELL_WIDTH * 0.46), CENTER_Y + 7),
                (x, board_top + 14),
            ]
        pygame.draw.polygon(surface, LIGHT_POINT if point % 2 == 0 else DARK_POINT, polygon)
        label_y = board_top + board_height - 25 if point < 12 else board_top + 14
        _center_text(surface, small, str(point + 1), (x, label_y), (252, 231, 177))

    bar_x = int(POINT_LEFT + (BAR_SLOT + 0.5) * CELL_WIDTH)
    pygame.draw.rect(surface, (121, 68, 43), (bar_x - 25, board_top + 8, 50, board_height - 16), border_radius=8)
    _center_text(surface, small, "BAR", (bar_x, CENTER_Y), (246, 216, 146))

    mask = env.action_mask() if not env.done else np.zeros(env.action_size, dtype=np.bool_)
    legal_sources = {
        decoded[0]
        for raw_action in np.flatnonzero(mask)
        if (decoded := decode_action(int(raw_action))) is not None
    }
    for source in legal_sources:
        if source == BAR:
            x = bar_x
            pygame.draw.rect(surface, HIGHLIGHT, (x - 29, board_top + 5, 58, board_height - 10), width=3, border_radius=8)
        else:
            x = point_center(source)
            y = board_top + board_height - 45 if source < 12 else board_top + 45
            pygame.draw.circle(surface, HIGHLIGHT, (x, y), 28, width=3)
    if selected_source is not None:
        x = bar_x if selected_source == BAR else point_center(selected_source)
        y = CENTER_Y if selected_source == BAR else (board_top + board_height - 45 if selected_source < 12 else board_top + 45)
        pygame.draw.circle(surface, ACCENT, (x, y), 32, width=4)

    # Checkers on points.  Only five are stacked visually; the number label
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
                y = board_top + board_height - 58 - stack_index * 35
            else:
                y = board_top + 58 + stack_index * 35
            _draw_checker(surface, pygame, (x, y), white)
        if count > 5:
            y = board_top + board_height - 58 if point < 12 else board_top + 58
            _center_text(surface, small, str(count), (x, y), BLACK_CHECKER if white else WHITE_CHECKER)

    # Bar stacks are drawn beside the BAR label.
    for player, count in ((WHITE, env.state.bar[WHITE]), (BLACK, env.state.bar[BLACK])):
        for index in range(min(count, 4)):
            y = CENTER_Y - 55 - index * 35 if player == WHITE else CENTER_Y + 55 + index * 35
            _draw_checker(surface, pygame, (bar_x, y), player == WHITE, radius=19)
        if count:
            _center_text(surface, small, f"{count}", (bar_x + 32, CENTER_Y - 55 if player == WHITE else CENTER_Y + 55), TEXT)


def _draw_sidebar(surface, pygame, env: BackgammonEnv, fonts, human_player: int, message: str, pending_source: int | None, restart_rect, pass_rect) -> None:
    small, normal, large = fonts
    panel_rect = (920, 30, 250, 740)
    pygame.draw.rect(surface, PANEL, panel_rect, border_radius=14)
    pygame.draw.rect(surface, (61, 76, 95), panel_rect, width=2, border_radius=14)
    _text(surface, large, "BG • RL", (945, 55), ACCENT)
    _text(surface, small, "BACKGAMMON", (946, 91), MUTED)

    current_name = "WHITE" if env.current_player == WHITE else "BLACK"
    current_color = WHITE_CHECKER if env.current_player == WHITE else (226, 139, 91)
    _text(surface, normal, f"Turn: {current_name}", (945, 130), current_color)
    if env.current_player == human_player:
        _text(surface, small, "You", (945, 161), ACCENT)
    else:
        _text(surface, small, "Neural agent", (945, 161), HIGHLIGHT)

    _text(surface, small, "Remaining dice", (945, 205), MUTED)
    dice = env.state.dice
    if dice:
        for index, die in enumerate(dice):
            rect = pygame.Rect(945 + (index % 2) * 45, 230 + (index // 2) * 45, 36, 34)
            pygame.draw.rect(surface, (239, 241, 231), rect, border_radius=7)
            _center_text(surface, normal, str(die), rect.center, (35, 43, 54))
    else:
        _text(surface, normal, "—", (945, 236), MUTED)

    _text(surface, small, "Borne off", (945, 335), MUTED)
    _text(surface, normal, f"White: {env.state.off[WHITE]:2d}", (945, 362), WHITE_CHECKER)
    _text(surface, normal, f"Black: {env.state.off[BLACK]:2d}", (945, 391), (226, 139, 91))
    _text(surface, small, f"Bar: W {env.state.bar[WHITE]}  B {env.state.bar[BLACK]}", (945, 428), MUTED)

    message_rect = pygame.Rect(940, 470, 210, 86)
    pygame.draw.rect(surface, (42, 51, 66), message_rect, border_radius=8)
    # Split long status text into two lines without requiring a text widget.
    words = message.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if normal.size(candidate)[0] > 190 and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for index, line in enumerate(lines[:3]):
        _text(surface, small, line, (950, 484 + index * 22), TEXT)

    pygame.draw.rect(surface, (55, 68, 84), restart_rect, border_radius=7)
    _center_text(surface, small, "R  New game", restart_rect.center, TEXT)
    pygame.draw.rect(surface, (55, 68, 84), pass_rect, border_radius=7)
    _center_text(surface, small, "SPACE  Pass", pass_rect.center, TEXT)

    if pending_source is not None:
        source_text = "BAR" if pending_source == BAR else f"point {pending_source + 1}"
        _text(surface, small, f"Choose die for {source_text}", (945, 660), HIGHLIGHT)


def run_game(
    checkpoint_path: str,
    human_player: int = WHITE,
    device_name: str = "auto",
    seed: int = 123,
) -> None:
    """Open the local GUI and run a human-vs-checkpoint game."""
    # Importing pygame only when the GUI is requested keeps training/evaluation
    # usable on headless machines.
    try:
        import pygame
    except ImportError as error:  # pragma: no cover - depends on local extras
        raise RuntimeError("pygame is not installed; run: python -m pip install pygame") from error

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

    restart_rect = pygame.Rect(940, 575, 210, 36)
    pass_rect = pygame.Rect(940, 615, 210, 36)
    pending_source: int | None = None
    message = "Click a highlighted checker to move."
    running = True
    last_ai_time = 0

    def restart() -> None:
        nonlocal pending_source, message, last_ai_time
        env.reset(starting_player=human_player)
        pending_source = None
        message = "Click a highlighted checker to move."
        last_ai_time = 0

    def apply_human_action(action: int) -> None:
        nonlocal pending_source, message
        try:
            _, _, done, info = env.step(action)
        except ValueError:
            message = "That move is not legal. Choose a highlighted option."
            pending_source = None
            return
        pending_source = None
        if done:
            winner = "White" if info["winner"] == WHITE else "Black"
            message = f"{winner} wins! Press R for a new game."
        elif env.current_player == human_player:
            message = "Your turn. Click a highlighted checker."
        else:
            message = "The neural agent is thinking..."

    while running:
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
                        apply_human_action(PASS_ACTION)
                    else:
                        message = "A move is available; passing is not legal."
                elif (
                    event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6)
                    and pending_source is not None
                    and not env.done
                    and env.current_player == human_player
                ):
                    die = event.key - pygame.K_0
                    action = (die - 1) * 25 + pending_source
                    if 0 <= action < PASS_ACTION and env.action_mask()[action]:
                        apply_human_action(action)
                    else:
                        message = "That die cannot move the selected checker."
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                if restart_rect.collidepoint(x, y):
                    restart()
                elif pass_rect.collidepoint(x, y) and not env.done and env.current_player == human_player:
                    if env.action_mask()[PASS_ACTION]:
                        apply_human_action(PASS_ACTION)
                    else:
                        message = "A move is available; passing is not legal."
                elif not env.done and env.current_player == human_player:
                    source = source_at_position(x, y)
                    if source is not None:
                        dice = _legal_dice_for_source(env.action_mask(), source)
                        if len(dice) == 1:
                            apply_human_action((dice[0] - 1) * 25 + source)
                        elif len(dice) > 1:
                            pending_source = source
                            message = "Press the die number on the keyboard."
                        else:
                            message = "That checker cannot move with the remaining dice."

        if not env.done and env.current_player != human_player:
            now = pygame.time.get_ticks()
            if now - last_ai_time >= 280:
                observation = torch.as_tensor(env.observation(), dtype=torch.float32, device=device)
                mask = torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device)
                with torch.no_grad():
                    action, _, _ = model.choose_action(observation, mask, deterministic=True)
                _, _, done, info = env.step(int(action.item()))
                last_ai_time = now
                if done:
                    winner = "White" if info["winner"] == WHITE else "Black"
                    message = f"{winner} wins! Press R for a new game."
                elif env.current_player == human_player:
                    message = "Your turn. Click a highlighted checker."

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
            restart_rect,
            pass_rect,
        )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
