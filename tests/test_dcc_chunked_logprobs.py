import copy
import importlib.util
import pathlib
import sys

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / "scripts"),
                str(pathlib.Path(__file__).resolve().parents[1] / "slime")]
from slime_plugins.dcc.hf_chunked_logprobs import completion_logprobs_chunked, resolve_logprob_backend
from slime_plugins.dcc.hf_policy import completion_logprobs


def model(lora=True):
    torch.manual_seed(59)
    config=Qwen3MoeConfig(vocab_size=97, hidden_size=32, intermediate_size=48,
        moe_intermediate_size=16,num_hidden_layers=2,num_attention_heads=4,
        num_key_value_heads=2,num_experts=4,num_experts_per_tok=2,
        max_position_embeddings=256,attention_dropout=0.0)
    base=Qwen3MoeForCausalLM(config)
    if lora:
        base=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.0,
            target_modules=["q_proj","v_proj","o_proj"],task_type="CAUSAL_LM"))
        # Exercise non-initial adapters too.
        with torch.no_grad():
            for name,p in base.named_parameters():
                if "lora_B" in name:p.normal_(0,0.02)
    return base


@pytest.mark.parametrize("lora,ckpt,chunk",[(False,False,1),(True,False,1),(True,False,3),(True,True,3)])
def test_real_qwen_values_and_gradients(lora,ckpt,chunk):
    original=model(lora); candidate=copy.deepcopy(original)
    if ckpt:
        for m in (original,candidate):
            m.enable_input_require_grads()
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    original.train();candidate.train()
    prompt=[2,7,11,15,27];response=[31,9,71,4,18,54,8]
    a=completion_logprobs(original,prompt,response)
    b=completion_logprobs_chunked(candidate,prompt,response,head_chunk_tokens=chunk)
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    weights=torch.arange(1,len(response)+1,dtype=torch.float32)
    loss_a=-(a*weights).sum();loss_b=-(b*weights).sum()
    loss_a.backward();loss_b.backward()
    named_b=dict(candidate.named_parameters())
    for name,p in original.named_parameters():
        q=named_b[name]
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=1e-6,msg=name)


def test_adapter_disabled_reference_and_eval():
    m=model(True);m.eval()
    with m.disable_adapter(),torch.no_grad():
        a=completion_logprobs(m,[2,3],[4,5,6]);b=completion_logprobs_chunked(m,[2,3],[4,5,6],head_chunk_tokens=2)
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    with torch.no_grad():
        enabled=completion_logprobs_chunked(m,[2,3],[4,5,6],head_chunk_tokens=2)
    assert not torch.allclose(a,enabled)


def test_forward_head_bound_and_backward_recompute():
    m=model(True);m.train();sizes=[]
    h=m.get_output_embeddings().register_forward_pre_hook(lambda module,args:sizes.append(args[0].shape[-2]))
    b=completion_logprobs_chunked(m,[2,3],[4,5,6,7,8,9,10],head_chunk_tokens=3)
    assert sizes==[3,3,1]
    (-b.mean()).backward()
    h.remove()
    assert len(sizes)>3 and max(sizes)==3


def test_guards():
    m=model()
    for chunk in (0,-1,True,1.5):
        with pytest.raises(ValueError):completion_logprobs_chunked(m,[1],[2],head_chunk_tokens=chunk)
    for prompt,response in (([],[2]),([1],[]),([True],[2]),([1],[-1])):
        with pytest.raises(ValueError):completion_logprobs_chunked(m,prompt,response)
    m.get_base_model().config.model_type="other"
    with pytest.raises(ValueError):completion_logprobs_chunked(m,[1],[2])


@pytest.mark.parametrize("field,value",[("alora_invocation_tokens",[7,11]),
    ("target_modules",{"q_proj"}), ("target_parameters",["experts.gate_up_proj"]),
    ("trainable_token_indices",[7]),("layer_replication",[(0,1)]),
    ("use_dora",True),("modules_to_save",["lm_head"])])
def test_nonstandard_peft_paths_rejected_before_forward(field,value):
    m=model(True)
    config=m.peft_config["default"]
    setattr(config,field,value)
    with pytest.raises(ValueError,match="standard q/v/o LoRA"):
        completion_logprobs_chunked(m,[2,7,11],[3,4])


def test_default_and_explicit_backend_recipes():
    from dcc_native_dpo_train import parser
    scorer, recipe = resolve_logprob_backend()
    assert scorer is completion_logprobs
    assert recipe == {"version": "dcc-logprob-backend-recipe-v1", "backend": "original",
                      "implementation_version": "hf-response-logits-fp32-raw-logprob-v1",
                      "head_chunk_tokens": None}
    assert parser().get_default("logprob_backend") == "original"
    assert parser().get_default("logprob_head_chunk_tokens") is None
    scorer, recipe = resolve_logprob_backend("qwen3-moe-chunked", 3)
    assert scorer.func is completion_logprobs_chunked
    assert scorer.keywords == {"head_chunk_tokens": 3}
    assert recipe["backend"] == "qwen3-moe-chunked" and recipe["head_chunk_tokens"] == 3
    assert resolve_logprob_backend("qwen3-moe-chunked")[1]["head_chunk_tokens"] == 128
    for backend, chunks in (("original", 3), ("other", None),
                            ("qwen3-moe-chunked", 0), ("qwen3-moe-chunked", True)):
        with pytest.raises(ValueError):
            resolve_logprob_backend(backend, chunks)


def test_chunked_whole_dpo_loop_records_recipe_and_matches_default(tmp_path):
    # Reuse the established synthetic protocol fixture, not production records.
    spec = importlib.util.spec_from_file_location("chunked_dpo_loop_fixture",
        pathlib.Path(__file__).with_name("test_dcc_native_dpo_train.py"))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    original = fixture.model()
    candidate = copy.deepcopy(original)
    pairs = fixture.pairs(1)
    for backend, value in (("original", original), ("qwen3-moe-chunked", candidate)):
        scorer, recipe = resolve_logprob_backend(backend, 2 if backend != "original" else None)
        source = {"source_commit": "a" * 40, "synthetic_cpu_only": True}
        output = tmp_path / backend
        fixture.run_dpo_loop(value, fixture.SavedTokenizer(), pairs, fixture.identity(),
            output=output, steps=1, pairs_per_step=1, logprob_fn=scorer,
            config={"scoring_backend": recipe, "training_source": source})
        config = __import__("json").loads((output / "final/training_config.json").read_text())
        assert config["scoring_backend"] == recipe and config["training_source"] == source
        assert config["reference"]["scoring_version"] == "raw-causal-full-completion-token-loglikelihood-sum-v1"
        assert config["sequence_score"] == "sum-all-actual-native-thought-close-JSON-EOS-token-logprobs"
    for name, tensor in fixture.parameters(original).items():
        torch.testing.assert_close(tensor, fixture.parameters(candidate)[name], atol=2e-9, rtol=2e-5)
    assert not torch.cuda.is_initialized()


def test_nf4_probe_comparator_detects_missing_keys_and_changed_gradients(tmp_path):
    # Fabricated CPU file-contract fixture ONLY; never evidence of actual NF4.
    import json
    from safetensors.torch import save_file
    from dcc_probe_chunked_logprobs import compare

    keys = {f"base_model.model.model.layers.0.self_attn.{projection}.lora_{matrix}.default.weight"
        for projection in ("q_proj", "v_proj", "o_proj") for matrix in ("A", "B")}
    gradients = {name: torch.full((2, 2), .01) for name in keys}
    common = {"source": {"source_commit": "a" * 40, "synthetic_cpu_only": True},
        "quantization": {"mode": "nf4", "expert_4bit_coverage": 1., "fabricated_cpu_fixture": True},
        "real_checkpoint_loaded": True, "prompt_ids": [1], "response_ids": [2, 3],
        "expected_gradient_shapes": {name: [2, 2] for name in keys},
        "adapter_seed": 59, "decoder_layers": 1, "lora_rank": 16, "lora_alpha": 32,
        "scores": [-3., -4.], "reference_scores": [-4., -4.5], "loss": 7.,
        "allocated_before": 1, "peak_allocated": 2, "peak_reserved": 3, "forward_backward_seconds": 1.}
    for backend in ("original", "qwen3-moe-chunked"):
        path = tmp_path / backend
        path.mkdir()
        recipe = resolve_logprob_backend(backend)[1]
        (path / "result.actual.json").write_text(json.dumps(dict(common, backend=backend, scoring_backend=recipe)))
        save_file(gradients, str(path / "adapter_initial.safetensors"))
        save_file(gradients, str(path / "gradients.safetensors"))
    result = compare(tmp_path)
    assert result["expected_gradient_key_count"] == 6
    assert result["gradient_max_abs_difference"] == result["loss_abs_difference"] == 0
    broken = dict(gradients)
    broken.pop(next(iter(keys)))
    target = tmp_path / "qwen3-moe-chunked/gradients.safetensors"
    save_file(broken, str(target))
    with pytest.raises(ValueError, match="Missing or extra"):
        compare(tmp_path)
    save_file({name: value * 3 for name, value in gradients.items()}, str(target))
    with pytest.raises(AssertionError):
        compare(tmp_path)
    assert not torch.cuda.is_initialized()
