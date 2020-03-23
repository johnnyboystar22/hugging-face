import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from transformers import BartForConditionalGeneration, BartTokenizer


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def generate_summaries(lns, out_file, batch_size=8, device=DEFAULT_DEVICE):
    fout = Path(out_file).open("w")
    model = BartForConditionalGeneration.from_pretrained("bart-large-cnn", output_past=True,).to(device)
    tokenizer = BartTokenizer.from_pretrained("bart-large")

    max_length = 140
    min_length = 55

    for batch in tqdm(list(chunks(lns, batch_size))):
        dct = tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt", pad_to_max_length=True)
        input_ids, attention_mask = trim_batch(
            dct["input_ids"].to(device), model.config.pad_token_id, attention_mask=dct["attention_mask"].to(device)
        )
        summaries = model.generate(
            input_ids=input_ids,
            attention_mask=input_ids,
            num_beams=4,
            length_penalty=2.0,
            max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=min_length + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=model.config.eos_token_id,
        )
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()


def _run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_path", type=str, help="like cnn_dm/test.source",
    )
    parser.add_argument(
        "output_path", type=str, help="where to save summaries",
    )
    parser.add_argument(
        "--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.",
    )
    parser.add_argument(
        "--bs", type=int, default=8, required=False, help="batch size: how many to summarize at a time",
    )
    args = parser.parse_args()
    lns = [" " + x.rstrip() for x in open(args.source_path).readlines()]
    generate_summaries(lns, args.output_path, batch_size=args.bs, device=args.device)


if __name__ == "__main__":
    _run_generate()
