"""
Tests for inference utilities in app/ml/export.py (Milestone 3).

Focus:
- Independent-heads legal-masking logic in predict_move (single-move edge cases, Black mover frame, promo).
- 100 random positions test for 100% legality guarantee.
- Opening book priority (model not called).
- Fallback model compatibility hook.
"""

import random
from unittest.mock import MagicMock, patch

import chess
import pytest
import torch

from app import config
from app.ml import export
from app.ml.model import ChessPolicyNet


@pytest.fixture
def untrained_model():
    """Create a lightweight model for inference testing."""
    return ChessPolicyNet(
        channels=16,
        num_res_blocks=1,
        hidden_dim=32,
        dropout=0.0,
    )


def test_predict_move_single_legal_move_white(untrained_model):
    """
    Hand-picked edge case 1: White has exactly ONE legal move.
    Position: White King on h1, Black King on f3, Black Queen on e2.
    Only legal move is h1g1.
    Test across a range of temperatures (deterministic 0.0 to high 5.0) and random seeds.
    """
    board = chess.Board("8/8/8/8/8/5k2/4q3/7K w - - 0 1")
    legal_moves = list(board.legal_moves)
    assert len(legal_moves) == 1
    expected_move = chess.Move.from_uci("h1g1")
    assert legal_moves[0] == expected_move

    for temp in [0.0, 0.1, 0.5, 1.0, 2.5, 5.0]:
        for seed in [1, 42, 999]:
            rng = random.Random(seed)
            move = export.predict_move(
                untrained_model,
                board,
                temperature=temp,
                rng=rng,
            )
            assert move == expected_move, (
                f"Expected single legal move {expected_move.uci()} but got {move.uci()} at temp={temp}"
            )


def test_predict_move_single_legal_move_black(untrained_model):
    """
    Hand-picked edge case 2: Black to move, exactly ONE legal move.
    Position: Black King on h8, White Queen on e7, White King on f6.
    Only legal move is h8g8.
    Verifies mover-frame mirroring and un-mirroring with independent-head masking.
    """
    board = chess.Board("7k/4Q3/5K2/8/8/8/8/8 b - - 0 1")
    legal_moves = list(board.legal_moves)
    assert len(legal_moves) == 1
    expected_move = chess.Move.from_uci("h8g8")
    assert legal_moves[0] == expected_move

    for temp in [0.0, 0.5, 1.0, 3.0]:
        for seed in [7, 77, 777]:
            rng = random.Random(seed)
            move = export.predict_move(
                untrained_model,
                board,
                temperature=temp,
                rng=rng,
            )
            assert move == expected_move, (
                f"Expected single legal move {expected_move.uci()} for Black but got {move.uci()} at temp={temp}"
            )


def test_predict_move_pinned_piece_edge_case(untrained_model):
    """
    Hand-picked edge case 3: Pinned piece.
    White Knight on f2 is pinned to King on h1 by Black Rook on a1? No, Bishop on c5 to g1.
    Let's use: White Kh1, White Nf2, Black Bc5. The knight on f2 is pinned along c5-g1 diagonal
    and cannot legally move! Only King has legal moves.
    """
    board = chess.Board("5r2/8/8/8/8/8/5N2/5K2 w - - 0 1")
    # Knight on f2 cannot move at all because King is on f1 and Rook is on f8.
    legal_moves = list(board.legal_moves)
    assert all(m.from_square != chess.F2 for m in legal_moves), "Pinned knight must have no legal moves"

    for seed in range(20):
        move = export.predict_move(
            untrained_model,
            board,
            temperature=1.0,
            rng=random.Random(seed),
        )
        assert move in board.legal_moves
        assert move.from_square != chess.F2, f"Pinned knight moved: {move.uci()}"


def test_predict_move_promotion_resolution(untrained_model):
    """
    Hand-picked edge case 4: Promotion resolution.

    Uses FEN "7k/4P3/8/8/8/8/8/4K3 w - - 0 1":
      - Black king on h8 is completely out of the pawn's path.
      - White pawn on e7 can freely promote to e8 (queen/rook/bishop/knight).
      - White king on e1 has five ordinary moves too.

    The fixture's validity is explicitly asserted (not assumed) before
    predict_move is called, so the test cannot silently pass on a broken FEN.

    Because temperature=0.0 the move is deterministic.  We unconditionally
    assert the returned move is legal.  When the pawn is chosen (from_sq==E7,
    to_sq==E8) the promotion piece must be one of the four legal classes.
    """
    board = chess.Board("7k/4P3/8/8/8/8/8/4K3 w - - 0 1")

    # --- Fixture validity assertions ---
    # Confirm the four promotion moves are genuinely legal in this position.
    assert chess.Move.from_uci("e7e8q") in board.legal_moves, "e7e8q must be legal"
    assert chess.Move.from_uci("e7e8r") in board.legal_moves, "e7e8r must be legal"
    assert chess.Move.from_uci("e7e8b") in board.legal_moves, "e7e8b must be legal"
    assert chess.Move.from_uci("e7e8n") in board.legal_moves, "e7e8n must be legal"

    move = export.predict_move(
        untrained_model,
        board,
        temperature=0.0,
    )

    # The returned move must always be legal (unconditional).
    assert move in board.legal_moves

    # Whenever the pawn is picked, the promotion piece must be valid.
    if move.from_square == chess.E7 and move.to_square == chess.E8:
        assert move.promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def test_predict_move_promotion_promo_logits_argmax(untrained_model):
    """
    Directly tests the 'multiple legal promo classes -> pick via promo_logits argmax' branch.

    Strategy
    --------
    1. Use the same FEN as test_predict_move_promotion_resolution so all four
       promotions are legal.
    2. Craft from/to logits so the pawn (E7→E8) wins the from/to selection at
       temperature=0.0 — give E7 a large from_sq logit and E8 a large to_sq
       logit while all king-move squares get 0.
    3. Set promo_logits[1] (KNIGHT, class 1) strictly highest among [0,1,2,3]
       so the argmax selects KNIGHT.
    4. Unconditionally assert the returned move has promotion == chess.KNIGHT,
       confirming the argmax-over-legal-promo-classes logic executed.

    This test also directly calls encoding.legal_promo_classes() to confirm it
    returns [0, 1, 2, 3] for the E7→E8 pair, validating that the multi-class
    branch is actually reachable before mocking anything.
    """
    from app.ml import encoding

    board = chess.Board("7k/4P3/8/8/8/8/8/4K3 w - - 0 1")

    # --- Verify that legal_promo_classes returns all four classes ---
    # White to move => mover frame == real frame => from_sq_mover=E7, to_sq_mover=E8
    promo_classes = encoding.legal_promo_classes(board, chess.E7, chess.E8)
    assert promo_classes == [0, 1, 2, 3], (
        f"Expected all four promo classes [0,1,2,3], got {promo_classes}"
    )

    # --- Build fake model output that forces E7→E8 (pawn) then KNIGHT ---
    import torch

    # from_sq logits: make E7 (index 52) dominant
    from_logits = torch.zeros(64)
    from_logits[chess.E7] = 100.0   # overwhelm any king-move from-square

    # to_sq logits: make E8 (index 60) dominant
    to_logits = torch.zeros(64)
    to_logits[chess.E8] = 100.0

    # promo_logits: make class 1 (KNIGHT) highest
    # [queen=0, knight=1, bishop=2, rook=3]
    promo_logits = torch.tensor([0.0, 10.0, 0.0, 0.0])

    mock_outputs = {
        "from_sq_logits": from_logits.unsqueeze(0),   # (1, 64)
        "to_sq_logits": to_logits.unsqueeze(0),        # (1, 64)
        "promo_logits": promo_logits.unsqueeze(0),     # (1, 4)
    }

    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = mock_outputs

    with patch.object(
        type(mock_model),
        "__call__",
        return_value=mock_outputs,
    ):
        # Patch model.eval() so it doesn't raise and the forward call returns mock_outputs
        mock_model.eval = MagicMock()

        with patch("torch.no_grad", return_value=torch.no_grad()):
            # We need the model(board=..., meta=...) call to return mock_outputs.
            # Use side_effect on the mock itself.
            mock_model.side_effect = None
            mock_model.return_value = mock_outputs

            move = export.predict_move(
                mock_model,
                board,
                temperature=0.0,
            )

    # The pawn promotion must have been chosen (from=E7, to=E8).
    assert move.from_square == chess.E7, (
        f"Expected pawn promotion from E7 but got from={chess.square_name(move.from_square)}"
    )
    assert move.to_square == chess.E8, (
        f"Expected pawn promotion to E8 but got to={chess.square_name(move.to_square)}"
    )
    # KNIGHT must be selected because promo_logits[1] was the highest.
    assert move.promotion == chess.KNIGHT, (
        f"Expected KNIGHT promotion (promo_logits[1] was highest) but got {move.promotion}"
    )


def test_predict_move_100_random_positions_100_percent_legal(untrained_model):
    """
    100 random legal positions must 100% produce legal moves.
    This is the core regression guard.
    """
    board = chess.Board()
    rng = random.Random(config.TRAIN_SEED)

    for i in range(100):
        if board.is_game_over():
            board = chess.Board()

        # Call predict_move
        move = export.predict_move(
            untrained_model,
            board,
            temperature=1.0,
            rng=rng,
        )
        assert move in board.legal_moves, (
            f"Step {i}: Returned illegal move {move.uci()} in position {board.fen()}"
        )
        # Advance board with the predicted move (or random if multiple)
        board.push(move)


def test_predict_move_opening_book_priority(untrained_model):
    """
    Opening book priority: if opening_book returns a match, predict_move returns
    that move and DOES NOT call the neural net.
    """
    mock_book = MagicMock()
    mock_book.query.return_value = {"e2e4": 1.0}
    mock_book.sample.return_value = chess.Move.from_uci("e2e4")

    board = chess.Board()

    # Pass a model whose forward pass would raise if called
    broken_model = MagicMock()
    broken_model.eval.side_effect = RuntimeError("Model should not be called when book matches!")

    move = export.predict_move(
        broken_model,
        board,
        opening_book=mock_book,
    )

    assert move == chess.Move.from_uci("e2e4")
    mock_book.query.assert_called_once_with(board)
    mock_book.sample.assert_called_once()
    broken_model.assert_not_called()


def test_predict_move_fallback_model_hook(untrained_model):
    """
    Verify fallback_model hook is accepted and does not crash.
    """
    board = chess.Board()
    move = export.predict_move(
        untrained_model,
        board,
        fallback_model="mock_base_model",
    )
    assert move in board.legal_moves


def test_save_and_load_model(untrained_model, tmp_path):
    """Test save_model and load_model state_dict roundtrip."""
    save_path = tmp_path / "model.pt"
    export.save_model(untrained_model, save_path)
    assert save_path.exists()

    loaded = export.load_model(
        save_path,
        channels=16,
        num_res_blocks=1,
        hidden_dim=32,
        dropout=0.0,
    )
    assert isinstance(loaded, ChessPolicyNet)

    # Check weights match
    for p1, p2 in zip(untrained_model.parameters(), loaded.parameters()):
        assert torch.equal(p1, p2)
