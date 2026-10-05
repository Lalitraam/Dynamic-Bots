"""
PyTorch Dataset for chess imitation learning.
Imports torch only in this file.
"""
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
import chess

from . import encoding
from ..services import classifier

class PlayerMoveDataset(Dataset):
    """
    Dataset that reads samples.parquet for a specific split and
    builds tensors on the fly for each row.
    
    Safe for num_workers=0 on Windows.
    """
    
    def __init__(self, parquet_path: str, split: str):
        """
        Parameters
        ----------
        parquet_path : str
            Path to the samples.parquet file.
        split : str
            One of 'train', 'val', or 'test'.
        """
        self.parquet_path = parquet_path
        self.split = split
        
        # We read the table into memory using PyArrow, filtering by split.
        # This is safe and fast enough for datasets of a few thousand games (~100k rows max).
        table = pq.read_table(
            parquet_path,
            filters=[('split', '=', split)]
        )
        # Convert to pandas for easier row-by-row iteration or keep as arrow?
        # PyArrow is fine, but pandas is easier to do .iloc.
        self.df = table.to_pandas()
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # 1. Setup board
        board = chess.Board(row["fen"])
        
        # 2. Board tensor
        board_array = encoding.encode_board(board)
        
        # 3. Move labels
        move = chess.Move.from_uci(row["uci"])
        move_dict = encoding.encode_move(board, move)
        
        # 4. Meta vector
        # The base time in seconds was saved as raw string in time_control, we need base_seconds
        parsed_tc = classifier.parse_time_control(row["time_control"])
        base_seconds = parsed_tc[0] if parsed_tc else 0
        inc_seconds = parsed_tc[1] if parsed_tc else 0
        
        # clock_before_move can be None/NaN. pandas converts None to NaN for float64.
        # We must handle NaN properly.
        import math
        cb = row["clock_before_move"]
        clock_before = None if math.isnan(cb) else float(cb)
        clock_available = not math.isnan(cb)
        
        meta_array = encoding.encode_meta(
            player_rating=row["player_rating"],
            opponent_rating=row["opponent_rating"],
            base_seconds=base_seconds,
            increment_seconds=inc_seconds,
            ply=row["ply"],
            clock_before=clock_before,
            clock_available=clock_available,
            halfmove_clock=row["halfmove_clock"],
        )
        
        # 5. Legal mask
        mask_array = encoding.legal_move_mask(board)
        
        # 6. Convert to PyTorch tensors
        # board: (18,8,8) uint8 -> float32 for NN input (often done in model, but we output uint8 here as per spec)
        # wait, spec says works with torch DataLoader, we can just return torch tensors with the right dtypes
        board_tensor = torch.from_numpy(board_array).to(torch.uint8)
        meta_tensor = torch.from_numpy(meta_array).to(torch.float32)
        mask_tensor = torch.from_numpy(mask_array).to(torch.bool)
        
        sample_weight = torch.tensor(row["sample_weight"], dtype=torch.float32)
        
        return {
            "board": board_tensor,
            "meta": meta_tensor,
            "from_sq": torch.tensor(move_dict["from_sq"], dtype=torch.int64),
            "to_sq": torch.tensor(move_dict["to_sq"], dtype=torch.int64),
            "promo": torch.tensor(move_dict["promo"], dtype=torch.int64),
            "piece_type": torch.tensor(move_dict["piece_type"], dtype=torch.int64),
            "is_capture": torch.tensor(move_dict["is_capture"], dtype=torch.float32), # or bool
            "is_check": torch.tensor(move_dict["is_check"], dtype=torch.float32),
            "is_castle": torch.tensor(move_dict["is_castle"], dtype=torch.float32),
            "legal_mask": mask_tensor,
            "sample_weight": sample_weight,
        }
