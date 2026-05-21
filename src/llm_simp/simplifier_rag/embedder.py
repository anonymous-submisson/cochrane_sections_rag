import logging
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Embedder:

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        pooling: str = "mean",
        device: Optional[str] = None,
    ):
        if pooling not in ("mean", "cls", "last"):
            raise ValueError(f"pooling must be 'mean', 'cls', or 'last', got {pooling}")
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(
            f"loading encoder: {model_name} (device={self.device}, pooling={pooling})"
        )
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = self.model.config.dim
        self.dim = int(hidden_size)
        logger.info(f"encoder loaded, embedding dim={self.dim}")

    @staticmethod
    def _mean_pool(
        last_hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @staticmethod
    def _last_pool(
        last_hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        return last_hidden[batch_idx, lengths]

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        embeddings = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            batch = [t if isinstance(t, str) and t.strip() else " " for t in batch]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            if self.pooling == "cls":
                pooled = out.last_hidden_state[:, 0, :]
            elif self.pooling == "last":
                pooled = self._last_pool(out.last_hidden_state, enc["attention_mask"])
            else:
                pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.detach().cpu().float().numpy())
            if show_progress:
                done = min(start + batch_size, total)
                logger.info(f"encoded {done}/{total}")
        return np.concatenate(embeddings, axis=0).astype(np.float32)


_DOI_PAIR_RE = re.compile(r"(CD\d+)", re.IGNORECASE)


def load_review_titles(data_json_path: str) -> Dict[str, str]:
    import json

    with open(data_json_path) as f:
        records = json.load(f)
    titles: Dict[str, str] = {}
    for rec in records:
        doi = rec.get("doi", "") or ""
        m = _DOI_PAIR_RE.search(doi)
        if not m:
            continue
        pair_id = m.group(1).upper()
        name = rec.get("name", "") or ""
        if name and pair_id not in titles:
            titles[pair_id] = name.strip()
    logger.info(f"loaded {len(titles)} review titles from {data_json_path}")
    return titles
