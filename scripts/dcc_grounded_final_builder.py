#!/usr/bin/env python3
"""Independent TRAIN gold-final SFT data; never an accepted native rollout.

The CLI imports the caller's frozen parent collection code. It never creates a
model or changes original selectors. ``check`` fully re-derives the dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

PROTOCOL = 'cuad-observed-gold-json-sft-v1'
MASK_VERSION = 'answer-json-and-eos-without-review-or-confidence-values-v1'
END_THINK, OPEN_THINK = 151668, 151667
SOURCE_PREFIXES = ('Initial contract retrieval (source material):\n',
                   'Tool observation (source material):\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':'))


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def normalized_map(text):
    """Whitespace-only normalization plus exact original character ranges."""
    chars, ranges = [], []
    previous_end = None
    for word in re.finditer(r'\S+', text):
        if chars:
            chars.append(' ')
            ranges.append((previous_end, word.start()))
        chars.extend(word.group())
        ranges.extend((index, index + 1) for index in range(word.start(), word.end()))
        previous_end = word.end()
    return ''.join(chars), ranges


def observed_chunks(document, effective_messages, decoded_original_prompt):
    """Evidence is a complete source message in P, never in thought H."""
    require(type(document) is str and document, 'Nonempty public document required')
    require(type(decoded_original_prompt) is str, 'Actual decoded P required')
    chunks = []
    for message_index, message in enumerate(effective_messages):
        if message.get('role') != 'user':
            continue
        content = message.get('content', '')
        prefix = next((value for value in SOURCE_PREFIXES if content.startswith(value)), None)
        if prefix is None:
            continue
        require(content in decoded_original_prompt,
                'Complete observed source message and offsets are absent from effective P')
        payload = content[len(prefix):].split('\nRemaining turns:', 1)[0]
        observation = strict_json(payload)
        require(type(observation) is dict, 'Source observation must be an object')
        if observation.get('error'):
            continue
        require(observation.get('document_length', len(document)) == len(document),
                'Observed document length differs from public source')
        hits = observation.get('hits', [observation])
        require(type(hits) is list, 'Observed hit list required')
        for hit_index, hit in enumerate(hits):
            require(type(hit) is dict, 'Observed hit must be an object')
            start, end, text = hit.get('start'), hit.get('end'), hit.get('text')
            require(type(start) is int and type(end) is int
                    and 0 <= start < end <= len(document) and type(text) is str
                    and document[start:end] == text,
                    'Source hit must equal document[start:end] exactly')
            chunks.append(dict(message_index=message_index, hit_index=hit_index,
                               start=start, end=end, text=text))
    require(chunks, 'No complete observed source message in effective P')
    return chunks


def merged_intervals(chunks, *, allow_contiguous_union):
    spans = sorted({(chunk['start'], chunk['end']) for chunk in chunks})
    if not allow_contiguous_union:
        return spans
    result = []
    for start, end in spans:
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def map_gold(document, gold, chunks, *, allow_contiguous_union=False):
    """Locate every gold span; targets are untouched continuous source strings.

    No case, punctuation, word-order or semantic normalization is allowed.
    Across multiple occurrences prefer exact gold bytes, then earliest offset.
    """
    require(type(gold) is list and all(type(span) is str and span.strip() for span in gold),
            'TRAIN gold must be an explicit list of nonempty text spans')
    for chunk in chunks:
        require(document[chunk['start']:chunk['end']] == chunk['text'],
                'Observed source offsets changed')
    intervals = merged_intervals(chunks, allow_contiguous_union=allow_contiguous_union)
    mappings = []
    for gold_index, span in enumerate(gold):
        needle, _ = normalized_map(span)
        candidates = {}
        for left, right in intervals:
            normalized, origins = normalized_map(document[left:right])
            offset = 0
            while needle:
                found = normalized.find(needle, offset)
                if found < 0:
                    break
                start = left + origins[found][0]
                end = left + origins[found + len(needle) - 1][1]
                target = document[start:end]
                require(normalized_map(target)[0] == needle,
                        'Whitespace mapping must preserve the complete normalized gold')
                candidates[start, end] = dict(gold_index=gold_index, gold=span,
                    document_start=start, document_end=end, target_text=target,
                    match='exact' if target == span else 'whitespace-offset-mapped',
                    source_intervals=[dict(message_index=c['message_index'], hit_index=c['hit_index'],
                        start=max(start, c['start']), end=min(end, c['end']))
                        for c in chunks if max(start, c['start']) < min(end, c['end'])])
                offset = found + 1
        if not candidates:
            return None
        chosen = min(candidates.values(), key=lambda value: (
            value['match'] != 'exact', value['document_start'], value['document_end']))
        chosen['matching_observed_occurrences'] = len(candidates)
        mappings.append(chosen)
    return mappings


def target_text(spans, *, final_review):
    # Review values and confidence values are teacher-forcing placeholders only.
    text = ('{"plan":"Review observed contract text.","checks":"Return only grounded spans.",'
            if final_review else '{')
    answer_start = len(text)
    text += '"answer":['
    confidence_ranges = []
    for index, span in enumerate(spans):
        if index:
            text += ','
        text += '{"text":' + json.dumps(span, ensure_ascii=False) + ',"confidence":'
        start = len(text)
        text += '0.9'
        confidence_ranges.append((start, len(text)))
        text += '}'
    text += ']}'
    return text, answer_start, confidence_ranges


def encode_target(tokenizer, spans, *, final_review, eos_ids):
    text, answer_start, ignored = target_text(spans, final_review=final_review)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded['input_ids']), list(encoded['offset_mapping'])
    require(ids and ids == tokenizer.encode(text, add_special_tokens=False)
            and len(ids) == len(offsets)
            and tokenizer.decode(ids, skip_special_tokens=False) == text,
            'Complete target must tokenize/decode exactly once with aligned offsets')
    require(type(tokenizer.eos_token_id) is int and tokenizer.eos_token_id in eos_ids,
            'Approved primary EOS required')
    require(not any(token in set(eos_ids) | {OPEN_THINK, END_THINK} for token in ids),
            'Target text contains a reserved native/EOS token')
    mask = []
    for start, end in offsets:
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(text),
                'Exact nonempty target character offsets required')
        mask.append(start >= answer_start and not any(
            start < right and end > left for left, right in ignored))
    require(any(mask), 'Target contains no supervised answer tokens')
    return dict(target_text=text, target_ids=ids + [tokenizer.eos_token_id],
                loss_mask=mask + [True], mask_version=MASK_VERSION)


def verify_conditioning(original_prompt_ids, thought_ids, phase2_prompt_ids, *, eos_ids):
    require(type(original_prompt_ids) is list and original_prompt_ids
            and type(thought_ids) is list and thought_ids,
            'Actual nonempty P and H token vectors required')
    require(thought_ids[-1] == END_THINK and thought_ids.count(END_THINK) == 1
            and not any(token in eos_ids for token in thought_ids),
            'H must retain its single actually generated natural close')
    require(phase2_prompt_ids == original_prompt_ids + thought_ids,
            'Conditioning must equal actual P + complete actual H without retemplating')


def import_parent(source_root):
    root = Path(source_root).resolve(strict=True)
    sys.path[:0] = [str(root / 'scripts'), str(root / 'slime')]
    names = ('dcc_collect_native_train', 'dcc_native_sft', 'dcc_native_physical',
             'dcc_cuad_retrieval', 'slime_plugins.dcc.hf_policy',
             'slime_plugins.dcc.metrics', 'slime_plugins.dcc.prompt_history')
    modules = [importlib.import_module(name) for name in names]
    for name, module in zip(names, modules):
        expected = root / ('slime' if name.startswith('slime_plugins.') else 'scripts')
        expected = expected / (name.replace('.', '/') + '.py' if name.startswith('slime_plugins.') else name + '.py')
        require(Path(module.__file__).resolve() == expected,
                'Parent imports differ from the explicitly supplied source root')
    return dict(zip(('collection', 'native', 'physical', 'retrieval', 'hf', 'metrics', 'history'), modules))


def effective_prompt(tokenizer, request, trace, identity, api):
    messages = request['messages']
    if identity.prompt_history is not None:
        history = api['history'].PromptHistoryConfig.from_dict(identity.prompt_history)
    else:
        history = None
    if history is not None and history.enabled:
        plan = api['history'].fit_prompt_history(tokenizer, messages,
            max_prompt_tokens=identity.max_prompt_tokens,
            max_new_tokens=identity.budget.total_tokens,
            max_model_len=identity.max_model_len, config=history)
        require(list(plan.prompt_ids) == trace['original_prompt_ids'],
                'Whole-turn effective prompt differs from actual P')
        messages = plan.messages
    else:
        require(api['hf'].encode_messages(tokenizer, messages, identity.max_prompt_tokens,
                    allow_truncation=False) == trace['original_prompt_ids'],
                'Effective prompt differs from actual P')
    return messages


def validate_parents(request, tokenizer, api):
    """All original closure and physical checks precede the first labels read."""
    from jsonschema import ValidationError
    collection = api['collection']
    profile, identity, schemas, receipt = collection.load_approved(request['profile'],
        request['parent_source_root'], request['parent_source_commit'], tokenizer)
    require(schemas.cuad is not None and schemas.cuad.retrieval is not None,
            'This protocol requires the task-bound CUAD collection profile')
    episodes, requests, traces = collection.read_closed(request['closed_collection'],
        profile=profile, receipt=receipt, world_size=request['collection_world_size'])
    closures = [strict_json((Path(request['closed_collection']) /
        f'generation_closed_rank{rank}.json').read_text())
        for rank in range(request['collection_world_size'])]
    require(all(row['collection_id'] == request['collection_id'] for row in closures),
            'Original collection ID differs from independently supplied binding')
    require(episodes and all(episode['benchmark'] == 'cuad' for episode in episodes),
            'This independent builder requires CUAD-only original closed journals')
    valid, audit, phase_owners = {}, [], set()
    for episode in episodes:
        public = api['retrieval'].public_cuad_task(episode['public_task'], task_id=episode['task_id'])
        parents = []
        for key in episode['trace_ids']:
            req, response = requests[key]
            try:
                require(key in traces and response['exception_type'] is None
                        and type(response['completion']) is dict,
                        'Original request lacks a complete successful physical response')
                trace = traces[key]
                schema = api['retrieval'].public_request_schema(public, req['messages'],
                    turn=req['turn_index'], max_steps=schemas.cuad.max_steps,
                    binding=schemas.cuad.retrieval)
                require(req['options'] == dict(max_new_tokens=identity.budget.total_tokens,
                    temperature=0., json_schema=schema), 'Original public request options differ')
                prior = [strict_json(traces[prior_key]['final_text']) for prior_key in
                         episode['trace_ids'][:req['turn_index']]]
                actual = [strict_json(message['content']) for message in req['messages']
                          if message['role'] == 'assistant']
                require(prior == actual, 'Historical assistant actions differ from actual prior traces')
                record = dict(source='cuad_train', split='train', task_id=episode['task_id'],
                              messages=copy.deepcopy(req['messages']), completion=trace['final_text'])
                collector = dict(split='train', task_input='public-train-task-without-gold',
                    gold_visible_to_policy=False, recorded_messages=True, synthetic_thought=False,
                    inserted_close=False, inserted_eos=False, episode_id=episode['episode_id'],
                    turn=req['turn_index'], rank=req['rank'], trace_id=key,
                    cuad_retrieval=copy.deepcopy(schemas.cuad.retrieval),
                    public_task=copy.deepcopy(episode['public_task']))
                physical = api['physical'].validate_native_physical(dict(contract=api['native'].CONTRACT,
                    record=record, identity=identity.manifest(tokenizer), collector=collector,
                    native_trace=trace), tokenizer, expected_identity=identity, original_schema=schema)
                completion = response['completion']
                require(completion == dict(text=trace['final_text'], prompt_ids=list(physical.prompt_ids),
                    completion_ids=list(physical.completion_ids), finish_reason='stop'),
                    'Original response differs from physically validated trace')
                for phase in trace['phases']:
                    phase_id = phase['request_id']
                    require(phase_id not in phase_owners, 'Original phase request reused across parents')
                    phase_owners.add(phase_id)
                messages = effective_prompt(tokenizer, req, trace, identity, api)
                decoded_p = tokenizer.decode(trace['original_prompt_ids'], skip_special_tokens=False)
                chunks = observed_chunks(public.document, messages, decoded_p)
                verify_conditioning(trace['original_prompt_ids'], trace['phases'][0]['token_ids'],
                    trace['phases'][1]['prompt_ids'], eos_ids=identity.eos_token_ids)
                parents.append(dict(request=req, trace=trace, chunks=chunks, schema=schema))
            except (ValueError, TypeError, KeyError, ValidationError) as error:
                audit.append(dict(task_id=episode['task_id'], trace_id=key,
                    disposition='original-parent-ineligible', reason=str(error),
                    original_episode_status=episode['status']))
        valid[episode['task_id']] = parents
    return episodes, valid, audit, profile, identity, receipt, closures, schemas


def derive(request, tokenizer, api):
    require(request.get('protocol') == PROTOCOL, 'Explicit independent SFT protocol required')
    require(type(request.get('allow_contiguous_union', False)) is bool,
            'Explicit boolean source union mode required')
    maximum = request.get('max_sequence_tokens', 53248)
    require(type(maximum) is int and maximum > 0, 'Positive complete sequence limit required')
    episodes, parents, audit, profile, identity, receipt, closures, schemas = validate_parents(request, tokenizer, api)
    # This is deliberately the first TRAIN labels access.
    labels = api['collection'].load_train_labels(request['train_labels'],
        {episode['task_id']: 'cuad' for episode in episodes})
    rows, overlength = [], []
    for episode in episodes:
        task_id, label = episode['task_id'], labels[episode['task_id']]
        gold = label['gold']
        selected = None
        for parent in parents[task_id]:
            mappings = map_gold(episode['public_task']['document'], gold, parent['chunks'],
                allow_contiguous_union=request.get('allow_contiguous_union', False))
            if mappings is not None:
                selected = parent, mappings
                break
        if selected is None:
            audit.append(dict(task_id=task_id, disposition='no-complete-grounded-physical-parent',
                positive=bool(gold), original_episode_status=episode['status']))
            continue
        parent, mappings = selected
        req, trace = parent['request'], parent['trace']
        spans = list(dict.fromkeys(value['target_text'] for value in mappings))
        answer = [dict(text=span, confidence=.9) for span in spans]
        score = api['metrics'].cuad_task_metrics(task_id, gold, answer)
        require(score['f1'] == 1. and score['fp'] == 0. and score['fn'] == 0.,
                'Mapped complete source target must pass actual CUAD TRAIN metric at F1=1')
        target = encode_target(tokenizer, spans,
            final_review=req['turn_index'] == schemas.cuad.max_steps - 1,
            eos_ids=identity.eos_token_ids)
        from jsonschema import Draft202012Validator
        Draft202012Validator(parent['schema']).validate(strict_json(target['target_text']))
        condition = list(trace['phases'][1]['prompt_ids'])
        sizes = dict(condition_tokens=len(condition), target_tokens=len(target['target_ids']),
                     total_tokens=len(condition) + len(target['target_ids']))
        if (sizes['condition_tokens'] > identity.max_prompt_tokens + identity.budget.thought_tokens
                or sizes['target_tokens'] > 4096 or sizes['total_tokens'] > maximum
                or sizes['total_tokens'] > identity.max_model_len):
            overlength.append(dict(task_id=task_id, **sizes))
            audit.append(dict(task_id=task_id, disposition='overlength-fatal-no-truncation', **sizes))
            continue
        rows.append(dict(protocol=PROTOCOL, origin='annotation-supervised-grounded-final',
            benchmark='cuad', split='train', task_id=task_id, category=label['category'],
            positive=bool(gold), gold=copy.deepcopy(gold), span_mappings=mappings,
            target_metric=score, **target, prompt_ids=condition, **sizes,
            parent=dict(episode_id=episode['episode_id'], episode_status=episode['status'],
                trace_id=trace['trace_id'], rank=trace['rank'], turn=req['turn_index'],
                original_stage=('final' if 'answer' in strict_json(trace['final_text']) else
                                strict_json(trace['final_text']).get('tool')),
                original_response_was_final='answer' in strict_json(trace['final_text']),
                phase1_request_id=trace['phases'][0]['request_id'],
                phase2_request_id=trace['phases'][1]['request_id'],
                original_prompt_ids=list(trace['original_prompt_ids']),
                actual_thought_ids=list(trace['phases'][0]['token_ids']),
                actual_final_ids=list(trace['phases'][1]['token_ids']),
                actual_final_text=trace['final_text']),
            original_prompt_and_thought_receive_loss=False,
            target_is_actual_native_generation=False,
            target_eos_is_synthetic_training_token=True,
            negative_supervision=('explicit-public-TRAIN-empty-annotation; observation-does-not-prove-absence'
                                  if not gold else None),
            actual_thought_semantic_consistency='unverified; original context preserved',
            original_native_selector_acceptance='not-asserted-or-modified'))
    counts = Counter(row['positive'] for row in rows)
    minimum_positive, minimum_negative = request.get('minimum_positive', 1), request.get('minimum_negative', 1)
    require(type(minimum_positive) is int and minimum_positive > 0
            and type(minimum_negative) is int and minimum_negative > 0,
            'Explicit positive per-polarity minimums required')
    manifest = dict(protocol=PROTOCOL, target_contract='synthetic-TRAIN-gold-JSON-with-real-masked-PH',
        native_only_training=False, training_method='supervised fine-tuning; not RL',
        mask_version=MASK_VERSION, selection='earliest physical request with all gold in effective P',
        allow_contiguous_union=request.get('allow_contiguous_union', False),
        original_journals_closed_before_labels=True, original_native_selector_modified=False,
        parent_source_receipt=receipt, parent_profile=profile, original_closures=closures,
        original_tasks=len(episodes), rows=len(rows), positive=counts[True], negative=counts[False],
        positive_categories=len({row['category'] for row in rows if row['positive']}),
        exclusions=len(audit), max_sequence_tokens=maximum, overlength=overlength,
        minimum_positive=minimum_positive, minimum_negative=minimum_negative,
        actual_max_sequence_tokens=max((row['total_tokens'] for row in rows), default=0),
        deficit=dict(positive=max(0, minimum_positive-counts[True]),
                     negative=max(0, minimum_negative-counts[False])),
        launchable=(not overlength and counts[True] >= minimum_positive and counts[False] >= minimum_negative),
        confidence_values_supervised=False, held_out_reads=False,
        real_model_or_GPU_called=False, benchmark_improvement_claimed=False)
    return rows, audit, manifest


def rows_bytes(rows):
    return ''.join(canonical(row) + '\n' for row in rows).encode('utf-8')


def digest(payload):
    # One small dataset pass; no checkpoint/model hashes or scans.
    return hashlib.blake2b(payload, digest_size=32).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'check'))
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args(argv)
    request = strict_json(args.request.read_text())
    api = import_parent(request['parent_source_root'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(request['tokenizer'], local_files_only=True,
                                              trust_remote_code=False)
    rows, audit, manifest = derive(request, tokenizer, api)
    payload = rows_bytes(rows)
    manifest['rows_blake2b_256'] = digest(payload)
    manifest['rows_bytes'] = len(payload)
    if args.command == 'build':
        args.dataset.mkdir(parents=True, exist_ok=False)
        (args.dataset / 'rows.jsonl').write_bytes(payload)
        (args.dataset / 'exclusions.jsonl').write_bytes(rows_bytes(audit))
        (args.dataset / 'manifest.json').write_text(canonical(manifest) + '\n')
    else:
        require((args.dataset / 'rows.jsonl').read_bytes() == payload,
                'Dataset rows differ from original closed-collection re-derivation')
        require((args.dataset / 'exclusions.jsonl').read_bytes() == rows_bytes(audit),
                'Dataset exclusions differ from original complete denominator')
        require(strict_json((args.dataset / 'manifest.json').read_text()) == manifest,
                'Dataset manifest differs from original closed-collection re-derivation')
    print(canonical(dict(protocol=PROTOCOL, command=args.command, rows=len(rows),
        positive=manifest['positive'], negative=manifest['negative'],
        positive_categories=manifest['positive_categories'], launchable=manifest['launchable'],
        overlength=overlength_count(manifest), deficit=manifest['deficit'])))
    return 0 if manifest['launchable'] else 2


def overlength_count(manifest):
    return len(manifest['overlength'])


if __name__ == '__main__':
    raise SystemExit(main())
