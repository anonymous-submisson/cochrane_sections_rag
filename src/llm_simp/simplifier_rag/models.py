import os
import re
import logging
from typing import List, Optional, Tuple, Union

import torch
from huggingface_hub import login
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SUPPORTED_MODELS = {
    "qwen-32b": "Qwen/Qwen2.5-32B-Instruct",
    "gemma-3-27b": "google/gemma-3-27b-it",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
}

DEFAULT_MODEL = "qwen-32b"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LLMSimplifier:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        seed: Optional[int] = None,
        max_model_len: Optional[int] = None,
        **kwargs,
    ):
        token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "HUGGING_FACE_TOKEN or HF_TOKEN environment variable not set"
            )

        login(token=token)

        if model_name in SUPPORTED_MODELS:
            model_path = SUPPORTED_MODELS[model_name]
            logger.info(f"resolved model alias '{model_name}' to '{model_path}'")
        else:
            model_path = model_name

        self.model_name = model_name
        self.model_path = model_path

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"loading model with vllm: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("tokenizer loaded for chat template")

        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
        )
        if seed is not None:
            llm_kwargs["seed"] = seed
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.model = LLM(**llm_kwargs)

        stop_tokens = [
            ">>>",
            "<<<",
            "Input:",
            "Note:",
            "Example",
            "\n\n\n",
            "Section text:",
        ]
        if "llama" in model_path.lower():
            stop_tokens.extend(["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"])
        if "gemma" in model_path.lower():
            stop_tokens.extend(["<end_of_turn>", "<start_of_turn>"])
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop_tokens,
            seed=seed,
        )

        logger.info("model loaded with vllm")

    def _format_prompt(self, prompt: Union[str, Tuple[str, str]]) -> str:
        if isinstance(prompt, tuple):
            system_prompt, user_prompt = prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted

    def _postprocess_output(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|[^|]+\|>", "", text)
        text = text.strip()

        if (
            re.fullmatch(r"\[[^\]]*\bDELETE\b[^\]]*\]\.?", text, flags=re.IGNORECASE)
            or text.upper() == "DELETE"
        ):
            return ""
        if (
            re.fullmatch(r"\[[^\]]*\bNONE\b[^\]]*\]\.?", text, flags=re.IGNORECASE)
            or text.upper() == "NONE"
        ):
            return ""

        cutoff_patterns = [
            "<<<",
            "Explanation:",
            "Note:",
            "---",
            "```",
            "COPY:",
            "REPHRASE:",
            "SPLIT:",
            "DELETE:",
        ]
        for pattern in cutoff_patterns:
            if pattern in text:
                text = text.split(pattern)[0]

        unwanted_starts = (
            "Answer:",
            "Simplified:",
            "Output:",
            "Here is",
            "Here's",
            "Sure,",
        )
        lines = text.strip().split("\n")
        filtered = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith(unwanted_starts):
                filtered.append(stripped)

        cleaned = " ".join(filtered)
        cleaned = re.sub(
            r"\[[^\]]*\b(delete|none|merge|copy|split|rephrase|ssplit|dsplit|ignore)\b[^\]]*\]",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\(note:.*?\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[一-鿿]+", "", cleaned)

        return cleaned.strip()

    def simplify_batch(self, prompts: List[Union[str, Tuple[str, str]]]) -> List[str]:
        logger.info(f"formatting {len(prompts)} prompts with chat template")
        formatted_prompts = [self._format_prompt(p) for p in prompts]

        logger.info(f"generating {len(prompts)} samples with vllm")
        outputs = self.model.generate(formatted_prompts, self.sampling_params)

        results = []
        for output in outputs:
            decoded = output.outputs[0].text
            results.append(self._postprocess_output(decoded))

        return results


def get_model(model_name: str, **kwargs) -> LLMSimplifier:
    return LLMSimplifier(model_name=model_name, **kwargs)
