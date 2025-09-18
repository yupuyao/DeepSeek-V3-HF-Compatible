import os
import json
import sys
from argparse import ArgumentParser
from typing import List, Iterator, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_file
from tqdm import tqdm
from model import Transformer, ModelArgs

def load_model_from_safetensors(model, ckpt_path, rank, world_size=8, prefix_to_remove="model."):
    with open(f"{ckpt_path}/model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    params_dict = dict(model.named_parameters())

    iterable = weight_map.items()
    if rank == 0:
        iterable = tqdm(weight_map.items(), desc="Loading weights", ncols=120)

    for name, shard_file in iterable:
        tensor_name = name[len(prefix_to_remove):] if name.startswith(prefix_to_remove) else name
        if tensor_name not in params_dict:
            continue

        shard_path = f"{ckpt_path}/{shard_file}"
        shard_tensor = load_file(shard_path)[name]

        param = params_dict[tensor_name]
        param_shape = param.shape
        shard_shape = shard_tensor.shape
        ndim = len(param_shape)

        if param_shape == shard_shape:
            param.data.copy_(shard_tensor)

        elif ndim == 1:
            dim0 = shard_shape[0]
            slice_size0 = dim0 // world_size
            start0 = rank * slice_size0
            end0 = start0 + slice_size0
            param.data.copy_(shard_tensor[start0:end0])

        elif ndim == 2:
            dim0, dim1 = shard_shape
            param0, param1 = param_shape

            slice_size0 = dim0 // world_size
            start0 = rank * slice_size0
            end0 = start0 + slice_size0
            slice_size1 = dim1 // world_size
            start1 = rank * slice_size1
            end1 = start1 + slice_size1

            if dim0 != param0 and dim1 == param1:
                param.data.copy_(shard_tensor[start0:end0, :])

            elif dim0 == param0 and dim1 != param1:
                param.data.copy_(shard_tensor[:, start1:end1])

            elif dim0 != param0 and dim1 != param1:
                param.data.copy_(shard_tensor[start0:end0, start1:end1])

    if rank == 0:
        print(f"Rank {rank}: weights loaded successfully")
    

    return model


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate_stream(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> Iterator[Tuple[List[List[int]], bool]]:
    """
    Generates new tokens with streaming support.
    
    Yields:
        Tuple[List[List[int]], bool]: (current_tokens, is_finished)
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        
        # Yield current completion tokens
        completion_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            current_completion = toks[prompt_lens[i]:cur_pos+1]
            # Remove padding tokens
            current_completion = [t for t in current_completion if t != -1]
            if eos_id in current_completion:
                current_completion = current_completion[:current_completion.index(eos_id)]
            completion_tokens.append(current_completion)
        
        yield completion_tokens, finished.all().item()
        
        if finished.all():
            break


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    # Use the streaming generator and return the final result
    final_tokens = None
    for tokens, is_finished in generate_stream(model, prompt_tokens, max_new_tokens, eos_id, temperature):
        final_tokens = tokens
        if is_finished:
            break
    return final_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    stream: bool = False,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
        stream (bool, optional): Whether to enable streaming output. Defaults to False.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    model = load_model_from_safetensors(model, ckpt_path, rank, world_size)

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            
            if stream and rank == 0:
                # Stream output
                print("", end="", flush=True)  # Prepare for streaming
                completion_text = ""
                for completion_tokens, is_finished in generate_stream(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature):
                    if completion_tokens[0]:  # If there are new tokens
                        new_completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
                        # Print only the new part
                        new_part = new_completion[len(completion_text):]
                        print(new_part, end="", flush=True)
                        completion_text = new_completion
                    if is_finished:
                        break
                print()  # New line after completion
                completion = completion_text
            else:
                # Non-stream output
                completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
                completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
                if rank == 0:
                    print(completion)
            
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.
        --stream (bool, optional): Enable streaming output. Defaults to False.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.stream)