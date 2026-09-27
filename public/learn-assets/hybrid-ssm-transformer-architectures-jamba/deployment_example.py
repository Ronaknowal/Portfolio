"""Optional large-model example, not executed by the lesson author.

Requires a compatible CUDA/PyTorch/Transformers/Accelerate installation and the
Mamba and causal-convolution kernels appropriate to it. Provision all model
weights, request caches and workspaces before attempting this program.

Example after resolving an actual revision:
  python deployment_example.py --model ai21labs/Jamba-v0.1 --revision COMMIT_HASH \
      --prompt-file prompt.txt

Replace COMMIT_HASH with the exact model repository revision you inspected.
Use --chat only for an instruction model whose tokenizer has a chat template.
"""
from pathlib import Path
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--prompt-file',type=Path,required=True)
    parser.add_argument('--chat',action='store_true')
    parser.add_argument('--max-new-tokens',type=int,default=64)
    args=parser.parse_args()
    if args.max_new_tokens<1:parser.error('Output token budget must be positive.')
    if not torch.cuda.is_available():parser.error('This example requires provisioned CUDA devices.')
    prompt=args.prompt_file.read_text(encoding='utf-8')
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,trust_remote_code=False)
    if args.chat:
        if not tokenizer.chat_template:parser.error('This tokenizer has no chat template.')
        prompt=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
            add_generation_prompt=True,tokenize=False)
    inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=not args.chat)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,
        trust_remote_code=False,dtype=torch.bfloat16,device_map='auto',use_mamba_kernels=True)
    limit=getattr(model.config,'max_position_embeddings',None)
    input_tokens=inputs['input_ids'].shape[1]
    if limit is not None and input_tokens+args.max_new_tokens>limit:
        parser.error(f'Input plus output budget exceeds the configured {limit} token limit.')
    device=model.get_input_embeddings().weight.device
    inputs={name:tensor.to(device) for name,tensor in inputs.items()}
    model.eval()
    padding_id=tokenizer.pad_token_id
    if padding_id is None:padding_id=tokenizer.eos_token_id
    with torch.inference_mode():
        generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,
            do_sample=False,use_cache=True,pad_token_id=padding_id)
    continuation=generated[0,input_tokens:]
    print(f'Input tokens: {input_tokens}; generated tokens: {len(continuation)}')
    print(tokenizer.decode(continuation,skip_special_tokens=True))

if __name__=='__main__':main()
