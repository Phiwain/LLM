"""Train a SentencePiece BPE tokenizer on bilingual EN+FR data."""
import argparse
import sentencepiece as spm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def train_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        normalization_rule_name="identity",
        input_sentence_size=2000000,
        shuffle_input_sentence=True,
    )
    print(f"Tokenizer saved to {model_prefix}.model and {model_prefix}.vocab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "tokenizer_sample.txt"),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=str(PROJECT_ROOT / "tokenizer" / "bpe"),
    )
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    train_tokenizer(
        input_file=args.input,
        model_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
    )
