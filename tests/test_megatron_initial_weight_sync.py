"""The first rollout weight sync must export actor, not auxiliary, weights.

Execute the real initialization and export methods with CPU synthetic tensors;
only external bootstrap, collectives, disk writer and Ray boundaries are stubbed.
"""
from __future__ import annotations

import ast
import copy
import logging
import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ACTOR_REL = Path("slime/slime/backends/megatron_utils/actor.py")

class Strip(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node.returns = None
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        return self.generic_visit(node)
    def visit_AnnAssign(self, node):
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)

def get_methods(source, class_name, names, target_name, bases):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    replacement = ast.ClassDef(name=target_name, bases=[ast.Name(id=b, ctx=ast.Load()) for b in bases], keywords=[], body=methods, decorator_list=[])
    return Strip().visit(replacement)

def compile_type(source, class_name, names, target_name, namespace, bases=()):
    node = get_methods(source, class_name, names, target_name, bases)
    mod = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(mod, str(ROOT / ACTOR_REL) + ':CPU-wiring-fixture', 'exec'), namespace)
    return namespace[target_name]

class Parent:
    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        self.args = args
        self.role = role
    def sleep(self):
        assert self._active_model_tag == 'actor'
        self.slept = True

class Backuper:
    @classmethod
    def create(cls, source_getter, single_tag):
        return cls(source_getter)
    def __init__(self, source_getter):
        self.source_getter = source_getter
        self.saved = {}
    @property
    def backup_tags(self):
        return self.saved.keys()
    def backup(self, tag):
        self.saved[tag] = {n: p.clone() for n, p in self.source_getter()}
    def restore(self, tag):
        for name, param in self.source_getter():
            param.copy_(self.saved[tag][name])
    def get(self, tag):
        return self.saved[tag]

noop = lambda *a, **kw: None
mpu = NS(get_virtual_pipeline_model_parallel_world_size=lambda: 1,
         get_data_parallel_world_size=lambda **kw: 1,
         get_context_parallel_world_size=lambda: 1)
dist = NS(get_rank=lambda: 0, barrier=noop)


def _run_case(source, transport, with_ref, with_teacher, offload, tmp_path):
    export_values = []
    # A one-element CPU model parameter, with independent synthetic actor/ref/teacher values.
    args = NS(debug_rollout_only=False, debug_train_only=False, distributed_timeout_minutes=1,
        num_gpus_per_node=1, hf_checkpoint='synthetic-only', offload_train=offload,
        keep_old_actor=False, vocab_size=None, model_name='synthetic', update_weight_mode='full',
        update_weight_transport=transport, colocate=False, rollout_data_postprocess_path=None,
        use_critic=False, use_fault_tolerance=False, update_weights_interval=1,
        ref_load='synthetic-reference', opd_teacher_load='synthetic-teacher', load='synthetic-actor',
        no_load_optim=False, no_load_rng=False, finetune=False, ref_ckpt_step=None,
        opd_teacher_ckpt_step=None, ckpt_step=None, custom_update_weight_post_write_path=None,
        update_weight_disk_dir=str(tmp_path / 'synthetic-disk'), update_weight_buffer_size=4096)
    live = torch.tensor([11.0], device='cpu')
    def load_checkpoint(model, *a, **kw):
        model[0].fill_({'synthetic-reference': 22.0, 'synthetic-teacher': 33.0,
                       'synthetic-actor': 11.0}[args.load])
        return None, None
    def save_hf_model_to_path(args, version_dir, model, **kw):
        export_values.extend(float(t.item()) for t in model)
    def named_params_and_buffers(args, model, **kw):
        return iter([('test.weight', model[0])])
    ns = dict(Parent=Parent, os=os, timedelta=timedelta, init=noop,
        monkey_patch_torch_dist=noop, register_default_process_group=noop,
        is_megatron_main_rank=lambda: False, init_tracking=noop,
        TrainProfiler=lambda _: NS(on_init_end=noop), dist=dist, get_gloo_group=lambda: None,
        AutoConfig=NS(from_pretrained=lambda *a, **kw: NS(vocab_size=64)),
        AutoTokenizer=NS(from_pretrained=lambda *a, **kw: NS(vocab_size=64)),
        initialize_model_and_optimizer=lambda *a: ([live], None, None, -1),
        mpu=mpu, TensorBackuper=Backuper, named_params_and_buffers=named_params_and_buffers,
        clear_memory=noop, load_checkpoint=load_checkpoint, nullcontext=nullcontext,
        torch_memory_saver=NS(disable=nullcontext), print_memory=noop,
        reload_process_groups=noop, destroy_process_groups=noop,
        logger=logging.getLogger('synthetic'), Path=Path, shutil=NS(rmtree=noop),
        save_hf_model_to_path=save_hf_model_to_path,
        all_gather_param=lambda name, param: param,
        convert_to_hf=lambda args, model_name, name, param, quant: [(name, param)],
        torch=torch)
    disk_src = (ROOT/'slime/slime/backends/megatron_utils/update_weight/update_weight_from_disk.py').read_text()
    disk = compile_type(disk_src, 'UpdateWeightFromDisk', ['__init__','update_weights'], 'Disk', ns)
    dist_src = (ROOT/'slime/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py').read_text()
    distributed = compile_type(dist_src, 'UpdateWeightFromDistributed', ['__init__','_iter_non_expert_chunks'], 'Distributed', ns)
    # Use the real distributed export iterator without networking or collectives.
    def send_nonexpert(self):
        self._is_pp_src_rank = True
        export_values.extend(float(t.item()) for chunk in self._iter_non_expert_chunks() for _, t in chunk)
    distributed.update_weights = send_nonexpert
    ns.update(UpdateWeightFromDisk=disk, UpdateWeightFromDistributed=distributed,
              UpdateWeightFromTensor=None)
    Actor = compile_type(source, 'MegatronTrainRayActor',
        ['init','load_other_checkpoint','_switch_model','update_weights'], 'Actor', ns, ('Parent',))
    actor = Actor()
    actor.init(args, 'actor', with_ref, with_teacher)
    init_active_tag = actor._active_model_tag
    # Execute the actual actor sync method with only the Ray manager boundary stubbed.
    manager_result = (['synthetic-engine'], None, 0, [1], [0], [{}])
    actor.rollout_manager = NS(get_updatable_engines_and_lock=NS(remote=lambda: manager_result))
    ns['ray'] = NS(get=lambda x: x)
    actor.update_weights()
    assert actor.weights_backuper.get('actor')['test.weight'].item() == 11.0
    assert args.load == 'synthetic-actor'  # checkpoint-loader argument restoration remains intact.
    return dict(transport=transport, with_ref=with_ref, with_teacher=with_teacher,
                offload=offload, init_active_tag=init_active_tag,
                exported=export_values, passed=(export_values == [11.0]))



@pytest.mark.parametrize("transport", ["disk", "nccl"])
@pytest.mark.parametrize("with_ref", [False, True])
@pytest.mark.parametrize("with_teacher", [False, True])
@pytest.mark.parametrize("offload", [False, True])
def test_initial_weight_sync_exports_actor(transport, with_ref, with_teacher, offload, tmp_path):
    result = _run_case(
        (ROOT / ACTOR_REL).read_text(), transport, with_ref, with_teacher, offload, tmp_path
    )
    assert result["exported"] == [11.0], result
    assert result["init_active_tag"] == "actor", result
