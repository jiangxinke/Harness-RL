"""Build a self-contained, escaped viewer from audited CUAD rollout records."""
from __future__ import annotations

import argparse
import collections
from datetime import datetime
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import quote, unquote, urlsplit


METHODS = {
    "grpo": ("Episode-only 对照", "按步骤访问计数归一化，并扣除无效动作罚分；不是标准的轨迹等权 GRPO。"),
    "gigpo": ("GiGPO", "整条轨迹项 EP ＋同状态步骤项 step。"),
    "graphgpo": ("GraphGPO", "整条轨迹项 EP ＋状态图距离产生的步骤项。"),
    "g2po": ("G2PO", "整条轨迹项 EP ＋节点项 NC ＋边项 EC。"),
    "graphpo": ("GraphPO（本次适配）", "逐层分支与语义合并；正确性项＋效率项，训练时共享前缀只计一次。"),
}
PHASES = {"training": "训练", "evaluation": "评测"}
COMPONENTS = {
    "episode_advantage": "EP：整条轨迹项",
    "local_advantage": "step：局部项",
    "graph_advantage": "Graph：图步骤项",
    "node_advantage": "NC：节点项",
    "edge_advantage": "EC：边项",
    "correctness_advantage": "正确性项",
    "efficiency_advantage": "效率项",
    "advantage": "实际合计",
}
DETAIL_COMPONENTS = {
    "discounted_return": "折扣回报", "graph_return": "图回报",
    "distance_target": "目标节点距离", "target_reachable": "图内可达成功节点",
    "node_value": "当前节点值", "next_value": "下一节点值", "edge_raw_gain": "边原始增益",
}


def esc(value):
    return html.escape(str(value), quote=True)


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def number(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def relative_url(path, output):
    return quote(Path(os.path.relpath(path, output)).as_posix(), safe="/")


def ident(method, phase, index):
    return f"{method}-{phase}-{index}"


def fail(condition, description):
    if not condition:
        raise ValueError(description)


def load_run(directory):
    audit_path = directory / "audit.json"
    audit = read_json(audit_path)
    method = audit["method"].lower()
    fail(method in METHODS, f"Unsupported method in {audit_path}: {method}")
    required_failures = [check for check in audit.get("checks", [])
                         if check.get("required") is True and check.get("status") != "passed"]
    fail(not required_failures, f"Required audit checks failed for {method}: {required_failures[:3]}")
    fail(audit.get("verdict") == "verified_record_consistency", f"Audit is not verified for {method}")
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else audit.get("manifest", {})
    config = audit.get("checkpoint", {}).get("training_config", {})
    phases, source_hashes = {}, {"audit.json": digest(audit_path)}
    for phase, filename, audit_key in (
        ("training", "train/rollouts_rank0.jsonl", "training_rollouts"),
        ("evaluation", "evaluation/rollouts.jsonl", "evaluation_rollouts"),
    ):
        source_path = directory / filename
        raw_records = read_jsonl(source_path)
        rows = audit[audit_key]
        fail(len(raw_records) == len(rows), f"{method} {phase}: raw/audit rollout count mismatch")
        indices = set()
        for row in rows:
            index, record = int(row["index"]), int(row["jsonl_record"])
            fail(index > 0 and index not in indices, f"{method} {phase}: invalid or duplicate display index")
            indices.add(index)
            fail(1 <= record <= len(raw_records), f"{method} {phase}: invalid JSONL record")
            raw = raw_records[record - 1]
            fail(row["task_id"] == raw["task_id"] and row["status"] == raw["status"],
                 f"{method} {phase} #{index}: task/status mismatch")
            fail(math.isclose(row["reward"], raw["reward"], abs_tol=1e-12),
                 f"{method} {phase} #{index}: reward mismatch")
            fail(row["answer"] == raw["answer"], f"{method} {phase} #{index}: answer mismatch")
            fail(len(row["steps"]) == len(raw["steps"]), f"{method} {phase} #{index}: decision count mismatch")
            assistant_messages = [message for message in raw["messages"] if message["role"] == "assistant"]
            fail(len(assistant_messages) == len(raw["steps"]), f"{method} {phase} #{index}: message count mismatch")
            for position, (step, sampled, message) in enumerate(zip(row["steps"], raw["steps"], assistant_messages)):
                fail(step["step"] == position and step["action_text"] == sampled["text"] == message["content"],
                     f"{method} {phase} #{index} step {position + 1}: action mismatch")
            row["_raw"] = raw
            row["_id"] = ident(method, phase, index)
        reported = audit[f"{phase}_statistics"]
        success = sum(row["status"] == "completed" and row["reward"] >= 1 - 1e-12 for row in rows)
        fail(reported["rollout_count"] == len(rows) and reported["success_count"] == success,
             f"{method} {phase}: aggregate success counts mismatch")
        tasks = {row["task_id"] for row in rows}
        fail(reported["distinct_tasks"] == len(tasks), f"{method} {phase}: task count mismatch")
        phases[phase] = rows
        source_hashes[filename] = digest(source_path)
    unique_steps = audit.get("unique_training_decisions", [])
    fail(len(unique_steps) == audit["training_statistics"]["unique_decisions"],
         f"{method}: unique training decisions mismatch")
    if method != "graphpo":
        fail(len(unique_steps) == sum(len(row["steps"]) for row in phases["training"]),
             f"{method}: ordinary trajectories unexpectedly share decisions")
    fail(sum(not step["is_action_valid"] for step in unique_steps) == audit["training_statistics"]["invalid_decisions"],
         f"{method}: invalid decision count mismatch")
    return dict(method=method, directory=directory, audit=audit, manifest=manifest, config=config,
                phases=phases, source_hashes=source_hashes)


def observation_view(observation):
    if not isinstance(observation, str):
        return "<pre>" + esc(dump(observation)) + "</pre>"
    value = None
    for start in (0, observation.find("\n") + 1):
        try:
            value = json.loads(observation[start:])
            break
        except (ValueError, TypeError):
            pass
    if not isinstance(value, dict):
        for marker in ("Initial contract retrieval (source material):\n", "Tool observation (source material):\n"):
            if observation.startswith(marker):
                try:
                    value, _ = json.JSONDecoder().raw_decode(observation[len(marker):])
                except ValueError:
                    pass
                break
    if not isinstance(value, dict):
        return "<pre>" + esc(observation) + "</pre>"
    pieces = []
    if value.get("error"):
        pieces.append('<p class="error">' + esc(value["error"]) + "</p>")
    hits = list(value.get("hits", []))
    if isinstance(value.get("text"), str):
        hits.append(value)
    for i, hit in enumerate(hits, 1):
        if not isinstance(hit, dict):
            pieces.append("<pre>" + esc(dump(hit)) + "</pre>")
            continue
        label = f"片段 {i} · 字符范围 {hit.get('start', '—')}–{hit.get('end', '—')}"
        if "score" in hit:
            label += f" · 检索分数 {number(hit['score'])}"
        pieces.append('<div class="source"><div class="source-label">' + esc(label)
                      + "</div><pre>" + esc(hit.get("text", dump(hit))) + "</pre></div>")
    pieces.append("<details><summary>原始观察数据</summary><pre>" + esc(observation) + "</pre></details>")
    return "".join(pieces)


def components_view(method, components):
    fields = {
        "grpo": ["episode_advantage", "advantage"],
        "gigpo": ["episode_advantage", "local_advantage", "advantage"],
        "graphgpo": ["episode_advantage", "graph_advantage", "advantage"],
        "g2po": ["episode_advantage", "node_advantage", "edge_advantage", "advantage"],
        "graphpo": ["correctness_advantage", "efficiency_advantage", "advantage"],
    }[method]
    if not components:
        return '<p class="muted">评测只生成回答，不计算训练优势。</p>'
    head = "".join("<th>" + esc(COMPONENTS[field]) + "</th>" for field in fields)
    cells = "".join("<td>" + esc(number(components.get(field))) + "</td>" for field in fields)
    more = [(label, components[field]) for field, label in DETAIL_COMPONENTS.items() if field in components]
    extra = ('<p class="mechanism">' + " · ".join(esc(label) + "：" + esc(number(value)) for label, value in more) + "</p>") if more else ""
    return '<div class="table-scroll"><table class="credit"><thead><tr>' + head + "</tr></thead><tbody><tr>" + cells + "</tr></tbody></table></div>" + extra


def after_messages(raw, step):
    positions = [i for i, message in enumerate(raw["messages"]) if message["role"] == "assistant"]
    end = positions[step + 1] if step + 1 < len(positions) else len(raw["messages"])
    return raw["messages"][positions[step] + 1:end]


def render_card(run, phase, row, output, links):
    method, raw, key = run["method"], row["_raw"], row["_id"]
    raw_path = output / "records" / (key + ".json")
    atomic_text(raw_path, dump(raw) + "\n")
    links.append(raw_path)
    label = METHODS[method][0]
    category = row["task_id"].rsplit("__", 1)[-1]
    success = row["status"] == "completed" and row["reward"] >= 1 - 1e-12
    title = f"{label} · {PHASES[phase]} #{row['index']:03d} · {category}"
    badge = "完整成功" if success else "未完整成功"
    pieces = [f'<details class="rollout" id="{esc(key)}" data-method="{esc(method)}" data-phase="{esc(phase)}" data-task="{esc(row["task_id"].lower())}">',
              '<summary><span class="card-title">' + esc(title) + '</span><span class="card-meta">'
              + esc(f"{len(row['steps'])} 次决策 · 任务得分 {number(row['reward'])} · {badge}") + "</span></summary>",
              '<div class="card-body"><p class="task-id">' + esc(row["task_id"]) + "</p>",
              '<p class="muted">' + esc(f"原始记录第 {row['jsonl_record']} 条；状态 {row['status']}。")]
    if "iteration" in row:
        pieces.append(esc(f" 采样轮次 {row['iteration']}，组内第 {row.get('episode_in_iteration', 0) + 1} 条。"))
    pieces += ['</p><p><a href="' + esc(relative_url(raw_path, output)) + '">查看这一条的完整原始 JSON（含 token / logprob）</a></p>',
               '<div class="result-grid"><div><b>最终回答</b><pre>' + esc(dump(row["answer"])) + "</pre></div>",
               '<div><b>评分分项</b><dl>' + "".join(
                   '<dt>' + esc(label) + '</dt><dd>' + esc(number(row["metrics"].get(field))) + '</dd>'
                   for field, label in [("f1", "条款匹配 F1"), ("observed_token_span_fraction", "已观察来源匹配"),
                                        ("verbatim_span_fraction", "旧逐字匹配诊断"), ("valid_format", "最终格式有效")])
               + "</dl></div></div>"]
    for step in row["steps"]:
        index = step["step"] + 1
        status = "通过输出与工具校验" if step["is_action_valid"] else "输出或工具未通过校验"
        flags = [status, f"新增来源片段 {step.get('new_source_texts', 0)}"]
        repeat = step.get("repeated_return_of_step")
        if repeat is not None:
            flags.append(f"反馈与第 {int(repeat) + 1} 步相同")
        pieces.append(f'<section class="decision" id="{esc(key)}-step-{index}"><h3>第 {index} 步</h3><p class="step-meta">'
                      + esc(" · ".join(flags)) + "</p>")
        pieces.append(components_view(method, step.get("components", {})))
        pieces.append('<h4>模型实际输出</h4><pre class="action">' + esc(step["action_text"]) + "</pre>")
        if step.get("errors"):
            pieces.append('<div class="errors"><b>本步错误反馈</b><ul>'
                          + "".join("<li>" + esc(error) + "</li>" for error in step["errors"]) + "</ul></div>")
        pieces.append('<details class="observation"><summary>行动前的最新观察</summary>'
                      + observation_view(step.get("observation", "")) + "</details>")
        feedback = after_messages(raw, step["step"])
        if feedback:
            pieces.append('<details class="feedback"><summary>行动后的完整反馈与控制提示</summary>')
            for message in feedback:
                pieces.append('<div class="message"><b>' + esc(message["role"]) + "</b>"
                              + observation_view(message["content"]) + "</div>")
            pieces.append("</details>")
        pieces.append('<details><summary>全部信用分项原值（内部编号沿用日志）</summary><pre>' + esc(dump(step.get("components", {}))) + "</pre></details></section>")
    pieces.append('<details class="full-messages"><summary>完整原始消息：系统提示、问题、全部材料与回答（'
                  + str(len(raw["messages"])) + " 条）</summary>")
    for index, message in enumerate(raw["messages"], 1):
        pieces.append('<div class="message"><h4>消息 ' + str(index) + " · " + esc(message["role"]) + "</h4><pre>"
                      + esc(message["content"]) + "</pre></div>")
    pieces.append("</details></div></details>")
    return "".join(pieces)


def render_cases(path, lookup):
    if path is None or not path.exists():
        return '<section class="intro"><h2>典型案例</h2><p>当前展示完整记录；典型案例说明尚未加入。</p></section>', 0
    value = read_json(path)
    cases = value["cases"] if isinstance(value, dict) else value
    fail(isinstance(cases, list), "selected_cases.json must contain a cases list")
    pieces = ['<section class="intro"><h2>典型案例</h2>']
    for case in cases:
        pieces.append('<article class="case"><h3>' + esc(case["title"]) + "</h3>")
        for field in ("explanation", "conclusion", "note"):
            if case.get(field):
                text = case[field] if isinstance(case[field], list) else [case[field]]
                pieces += ["<p>" + esc(paragraph) + "</p>" for paragraph in text]
        refs = []
        for ref in case.get("refs", []):
            method, phase, index = ref["method"].lower(), ref.get("phase", "training"), int(ref["rollout"])
            key = ident(method, phase, index)
            fail(key in lookup, f"Case refers to an absent rollout: {key}")
            caption = f"{METHODS[method][0]} · {PHASES[phase]} #{index:03d}"
            refs.append('<a href="#' + esc(key) + '">' + esc(caption) + "</a>")
            for step in ref.get("steps", []):
                fail(type(step) is int and 1 <= step <= len(lookup[key]["steps"]), f"Case has invalid step: {key} {step}")
                refs.append('<a href="#' + esc(key) + "-step-" + str(step) + '">第 ' + str(step) + " 步</a>")
        pieces.append('<p class="case-links">' + " · ".join(refs) + "</p></article>")
    pieces.append("</section>")
    return "".join(pieces), len(cases)


CSS = """
:root{color-scheme:light;--ink:#17202b;--muted:#586575;--border:#dce2e9;--accent:#215da6;--paper:#fff;--back:#f4f6f9}
*{box-sizing:border-box}body{margin:0;background:var(--back);color:var(--ink);font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif}
main{max-width:1280px;margin:auto;padding:38px 28px 80px}h1{font-size:32px;line-height:1.25;margin:0 0 12px}h2{font-size:22px;margin:0 0 14px}h3{font-size:18px;margin:0 0 8px}h4{font-size:15px;margin:15px 0 8px}p{margin:8px 0 14px}a{color:var(--accent);text-underline-offset:3px}a:hover{color:#173e71}
.intro{background:var(--paper);border:1px solid var(--border);border-radius:12px;padding:24px;margin:20px 0}.muted,.step-meta,.mechanism{color:var(--muted);font-size:14px}.notice{border-left:4px solid #d3a23f;padding-left:14px}.table-scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:12px 0}th,td{text-align:left;border-bottom:1px solid var(--border);padding:10px 12px;vertical-align:top}th{font-size:14px;background:#eef2f7;white-space:nowrap}td{font-size:14px}.credit{max-width:850px}.credit td{font:600 15px ui-monospace,Consolas,monospace}.credit td:last-child{color:#164d8d;background:#f0f6ff}
.toolbar{position:sticky;top:0;z-index:5;background:#fffffff5;backdrop-filter:blur(8px);border:1px solid var(--border);border-radius:10px;padding:14px 18px;display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:24px 0}label{display:flex;flex-direction:column;font-size:13px;gap:4px}input,select,button{font:inherit;padding:8px 11px;border:1px solid #bec9d7;border-radius:6px;background:#fff;color:var(--ink)}input{width:280px}button{cursor:pointer;font-size:14px}button:hover{background:#eef3fa}.count{font-size:13px;color:var(--muted);margin-left:auto}
.rollout{border:1px solid var(--border);border-radius:10px;background:#fff;margin:12px 0;overflow:hidden}.rollout>summary{display:flex;justify-content:space-between;gap:20px;cursor:pointer;padding:18px 20px;background:#fff}.rollout>summary:hover{background:#f8faff}.rollout[open]>summary{border-bottom:1px solid var(--border);background:#f8faff}.card-title{font-weight:650}.card-meta{font-size:13px;color:var(--muted);white-space:nowrap}.card-body{padding:18px 24px}.task-id{font-family:ui-monospace,Consolas,monospace;font-size:13px;overflow-wrap:anywhere}.result-grid{display:grid;grid-template-columns:2fr 1fr;gap:20px}.result-grid>div{min-width:0}dl{display:grid;grid-template-columns:1fr auto;font-size:14px;gap:8px;margin:10px 0}dd{margin:0;font-family:ui-monospace,Consolas,monospace}
.decision{border-top:1px solid var(--border);margin-top:24px;padding-top:22px;scroll-margin-top:160px}.decision>details,.full-messages{margin:12px 0;border:1px solid var(--border);border-radius:7px;padding:10px 14px}details>summary{cursor:pointer;font-weight:550}.full-messages{background:#f8f9fc}.source{margin:12px 0;border-left:3px solid #c9d7e8;padding-left:12px}.source-label{font-size:12px;color:var(--muted)}pre{font:13px/1.7 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;background:#f5f7fa;padding:12px;border-radius:5px;max-height:620px;overflow:auto;margin:8px 0}.action{background:#eff5fc;max-height:400px}.error,.errors{color:#973d2b}.errors{background:#fff6f2;border:1px solid #eed6cb;border-radius:6px;padding:12px 16px;font-size:14px}.errors ul{margin:6px 0 0;padding-left:22px}.message{margin:16px 0}.message b{font-size:13px}.case{border-top:1px solid var(--border);padding:16px 0}.case:last-child{padding-bottom:0}.case-links{font-size:14px}.empty{padding:36px;text-align:center;color:var(--muted)}footer{font-size:13px;color:var(--muted);margin-top:28px}[hidden]{display:none!important}
@media(max-width:800px){main{padding:20px 12px}.intro{padding:16px}.rollout>summary{display:block;padding:14px}.card-meta{display:block;margin-top:5px;white-space:normal}.card-body{padding:12px}.result-grid{grid-template-columns:1fr}.toolbar{position:static}input{width:230px}h1{font-size:25px}.count{margin-left:0}.decision{scroll-margin-top:20px}}
"""
JS = """
const cards=[...document.querySelectorAll('.rollout')];
const method=document.getElementById('method'),phase=document.getElementById('phase'),query=document.getElementById('query');
function applyFilters(){let count=0;const q=query.value.toLowerCase().trim();for(const card of cards){const show=(!method.value||card.dataset.method===method.value)&&(!phase.value||card.dataset.phase===phase.value)&&(!q||card.dataset.task.includes(q));card.hidden=!show;if(show)count++;}document.getElementById('visible-count').textContent=`显示 ${count} / ${cards.length} 条`;document.getElementById('empty').hidden=count!==0;}
for(const input of [method,phase,query])input.addEventListener('input',applyFilters);
document.getElementById('open-visible').addEventListener('click',()=>{for(const card of cards)if(!card.hidden)card.open=true;});
document.getElementById('close-visible').addEventListener('click',()=>{for(const card of cards)if(!card.hidden)card.open=false;});
function revealHash(){if(!location.hash)return;const target=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(!target)return;const card=target.closest('.rollout');if(card){method.value='';phase.value='';query.value='';applyFilters();card.open=true;}let node=target.parentElement;while(node){if(node.tagName==='DETAILS')node.open=true;node=node.parentElement;}setTimeout(()=>target.scrollIntoView({block:'start'}),0);}
addEventListener('hashchange',revealHash);applyFilters();revealHash();
"""


class LinkInspector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.external_dependencies = [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if "href" in attrs:
            self.links.append(attrs["href"])
        if tag in {"script", "link", "img", "iframe"} and (attrs.get("src") or attrs.get("href")):
            self.external_dependencies.append((tag, attrs.get("src") or attrs.get("href")))


def check_javascript(path):
    node = shutil.which("node")
    if node is None:
        return {"status": "not_run", "reason": "Node.js unavailable"}
    test = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[1],'utf8');
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];assert.equal(scripts.length,1);
const cards=[...html.matchAll(/<details class="rollout" id="([^"]+)" data-method="([^"]+)" data-phase="([^"]+)" data-task="([^"]*)">/g)].map(m=>({id:m[1],dataset:{method:m[2],phase:m[3],task:m[4]},hidden:false,open:false,parentElement:null}));
const ids={};for(const id of ['method','phase','query','visible-count','empty','open-visible','close-visible'])ids[id]={value:'',hidden:false,textContent:'',handlers:{},addEventListener(k,f){this.handlers[k]=f;}};
const location={hash:''};const listeners={};let scrolled=false;
const targetCard=cards[0],target={closest(){return targetCard;},parentElement:targetCard,scrollIntoView(){scrolled=true;}};targetCard.tagName='DETAILS';
const document={querySelectorAll(){return cards;},getElementById(id){return id==='test-target'?target:ids[id];}};
vm.runInNewContext(scripts[0][1],{document,location,addEventListener(k,f){listeners[k]=f;},setTimeout(f){f();}});
assert.equal(cards.filter(c=>!c.hidden).length,cards.length);
ids.method.value=targetCard.dataset.method;ids.phase.value='evaluation';ids.method.handlers.input();
const evaluation=cards.filter(c=>!c.hidden);assert(evaluation.length>0);assert(evaluation.every(c=>c.dataset.phase==='evaluation'&&c.dataset.method===targetCard.dataset.method));
ids['open-visible'].handlers.click();assert(evaluation.every(c=>c.open));assert(cards.filter(c=>c.hidden).every(c=>!c.open));
ids['close-visible'].handlers.click();assert(evaluation.every(c=>!c.open));
ids.query.value='this task must not exist 123456';ids.query.handlers.input();assert(cards.every(c=>c.hidden));assert.equal(ids.empty.hidden,false);
location.hash='#test-target';listeners.hashchange();assert.equal(cards.filter(c=>!c.hidden).length,cards.length);assert(targetCard.open&&scrolled);
console.log(JSON.stringify({status:'passed',cards:cards.length,method_phase_filter_count:evaluation.length,checks:['syntax','method_phase_filter','keyword_empty_state','expand_visible_only','collapse_visible','case_anchor_reveal']}));
'''
    result = subprocess.run([node, "-e", test, str(path)], check=True, text=True, capture_output=True, encoding="utf-8")
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/credit-rebuild-20261029/comparison"))
    parser.add_argument("--selected-cases", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = [load_run(directory.resolve()) for directory in args.run_dir]
    methods = [run["method"] for run in runs]
    fail(len(set(methods)) == len(methods), "Each method may be supplied only once")
    lookup = {row["_id"]: row for run in runs for rows in run["phases"].values() for row in rows}
    case_path = args.selected_cases.resolve() if args.selected_cases else output / "selected_cases.json"
    cases, case_count = render_cases(case_path, lookup)
    missing = [METHODS[name][0] for name in METHODS if name not in methods]
    evaluation_sets = [{row["task_id"] for row in run["phases"]["evaluation"]} for run in runs]
    evaluation_counts = [collections.Counter(row["task_id"] for row in run["phases"]["evaluation"]) for run in runs]
    common_evaluation = all(tasks == evaluation_sets[0] for tasks in evaluation_sets)
    repeat_values = {count for counts in evaluation_counts for count in counts.values()}
    if common_evaluation and len(repeat_values) == 1:
        task_count, repeats = len(evaluation_sets[0]), next(iter(repeat_values))
        evaluation_note = f"评测只有 {task_count} 个独立合同问题，每题重复 {repeats} 次，不是 {task_count * repeats} 个独立问题。"
    else:
        evaluation_note = "各方法的评测任务或重复次数不完全一致，请结合原始设置阅读；重复轨迹不是独立合同问题。"
    condition_rows, statistics_rows, summaries = [], [], []
    links = []
    for run in runs:
        method, config, audit = run["method"], run["config"], run["audit"]
        manifest_link = run["directory"] / "manifest.json"
        links.append(manifest_link)
        commit = run["manifest"].get("source_commit", "未记录")
        condition_rows.append("<tr><td><b>" + esc(METHODS[method][0]) + "</b><br>" + esc(METHODS[method][1])
            + "</td><td>" + esc(config.get("cuad_protocol", run["manifest"].get("protocol", "未记录")))
            + "</td><td>" + esc((f"最多 {config.get('max_steps', '—')} 次决策；分支配置 {config.get('graphpo_branching', '—')}，叶路径数按实际记录统计" if method == "graphpo" else f"最多 {config.get('max_steps', '—')} 次决策；每组 {config.get('group_size', '—')} 条"))
            + "</td><td><a href=\"" + esc(relative_url(manifest_link, output)) + "\">" + esc(commit[:10]) + " · 原始设置</a></td></tr>")
        train, evaluation = audit["training_statistics"], audit["evaluation_statistics"]
        statistics_rows.append("<tr><td>" + esc(METHODS[method][0]) + "</td>"
            + "".join("<td>" + esc(value) + "</td>" for value in (
                f"{train['success_count']}/{train['rollout_count']}",
                f"{evaluation['success_count']}/{evaluation['rollout_count']}",
                f"{train['distinct_tasks']} / {evaluation['distinct_tasks']}",
                train["unique_decisions"], f"{train['invalid_decisions']}/{train['unique_decisions']}",
                audit.get("checkpoint", {}).get("optimizer_steps", "—"))) + "</tr>")
        summaries.append({"method": method, "label": METHODS[method][0], "variant": METHODS[method][1],
                          "run_dir": str(run["directory"]), "source_commit": commit,
                          "training": train, "evaluation": evaluation,
                          "optimizer_steps": audit.get("checkpoint", {}).get("optimizer_steps"),
                          "costs": audit.get("costs", {}), "source_sha256": run["source_hashes"]})
    cards = "".join(render_card(run, phase, row, output, links)
                    for run in runs for phase in PHASES for row in run["phases"][phase])
    options = "".join('<option value="' + esc(run["method"]) + '">' + esc(METHODS[run["method"]][0]) + "</option>" for run in runs)
    coverage = f"当前已载入 {len(runs)} 组实验、{len(lookup)} 条训练与评测记录。"
    if missing:
        coverage += "尚未纳入本页：" + "、".join(missing) + "。"
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    page += '<title>CUAD 基线轨迹对照</title><style>' + CSS + '</style></head><body><main>'
    page += '<header><h1>CUAD 基线轨迹对照</h1><p>' + esc(coverage) + '</p><p class="muted">先看每一步做了什么、拿到了什么材料，再看信用信号如何分配。</p></header>'
    page += '<section class="intro"><h2>实验条件与方法</h2><p>策略模型为 Qwen2.5-3B-Instruct，使用 DCC EpisodeRunner 合同工具流程和 LoRA 更新，未运行原生 SP 中枢与子 Agent。任务得分为本次适配的条款匹配 F1 与已观察来源匹配，不是 CUAD 官方完整排行榜指标。完整成功才提供成功终局信号：GiGPO、GraphGPO、G2PO 与 episode-only 对照用 10/0，GraphPO 的正确性计算用 1/0。</p>'
    if "graphpo" in methods:
        page += '<p>GraphPO 按论文实现；尚未找到官方公开代码。其状态摘要使用可追溯摘录，辅助编码器使用 GTE ModernBERT base，替代论文的 SFR-Embedding-2_R。这些适配会影响状态合并，相关观察只适用于当前配置。</p>'
    page += '<div class="table-scroll"><table><thead><tr><th>方法与计算范围</th><th>协议版本</th><th>采样上限</th><th>代码与设置</th></tr></thead><tbody>' + "".join(condition_rows) + '</tbody></table></div></section>'
    page += cases
    page += '<section class="intro"><h2>核心统计</h2><p class="notice">这些小样本数字不是方法排名。各方法更新后采样路径不同；相同题目与种子不能把最终分差变成单一机制的因果结论。' + esc(evaluation_note) + '</p>'
    page += '<div class="table-scroll"><table><thead><tr><th>方法</th><th>训练完整成功</th><th>评测完整成功</th><th>训练题 / 评测题</th><th>实际训练决策数</th><th>未通过动作校验</th><th>参数更新次数</th></tr></thead><tbody>' + "".join(statistics_rows) + '</tbody></table></div>'
    page += '<p class="muted">“完整成功”要求任务得分达到 1。GraphPO 共享前缀在实际训练决策数中只计一次。部分正确仍显示原始 F1；无效动作罚分、来源匹配及协议失败应与信用方法分别解释。参数更新也可能含 KL 项，不等于每轮都有任务优势信号。</p></section>'
    graphpo_run = next((run for run in runs if run["method"] == "graphpo"), None)
    if graphpo_run:
        graph_rounds = [row for row in read_jsonl(graphpo_run["directory"] / "train/train_rank0.jsonl")
                        if row.get("stage") == "graphpo"]
        nonzero_rounds = sum(row.get("nonzero_policy_advantages", 0) > 0 for row in graph_rounds)
        zero_updates = sum(row.get("optimizer_step_applied", False) and row.get("nonzero_policy_advantages", 0) == 0 for row in graph_rounds)
        graph_costs = [row["costs"] for row in graphpo_run["audit"]["costs"]["graph_auxiliary_costs"]]
        summary_calls = sum(row.get("summarizer_usage", {}).get("calls", 0) for row in graph_costs)
        summary_inputs = sum(row.get("summarizer_usage", {}).get("input_tokens", 0) for row in graph_costs)
        actor_inputs = graphpo_run["audit"]["costs"]["generated_prompt_tokens"]
        page += '<section class="intro"><h2>GraphPO 还花了哪些成本</h2><p>' + esc(f"实际构建 {len(graph_rounds)} 张图，{nonzero_rounds} 轮出现非零任务优势；另有 {zero_updates} 次参数更新没有非零任务优势，不能把更新次数全算成任务学习信号。") + '</p><p>' + esc(f"除 Actor 的 {actor_inputs:,} 个输入 token 外，还调用状态摘要 {int(summary_calls)} 次，增加 {int(summary_inputs):,} 个输入 token，并执行相应编码。这是当前适配的实际额外成本，不是原论文实现的速度对比。") + '</p></section>'
    paired_path = output.parent / "paired-credit-abbb72c-v2.json"
    if paired_path.exists():
        paired = json.loads(paired_path.read_text(encoding="utf-8"))
        fail(paired.get("parameters_updated") is False, "Paired diagnostic unexpectedly updated model parameters")
        paired_rows = []
        for key, label in (("current_observation", "只看当前观察"),
                           ("plus_remaining_budget", "再区分剩余步数"),
                           ("plus_budget_and_visible_source_coverage", "再区分已见合同内容")):
            values = paired["variants"][key]["gigpo"]
            paired_rows.append("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in
                (label, values["groups"], values["singleton_groups"], values["zero_local"],
                 values["invalid_positive_local"], values["invalid_positive_total"])) + "</tr>")
        page += '<section class="intro"><h2>把状态分得更细，会怎样？</h2><p>固定 GiGPO 采到的同一批 96 条轨迹、200 次决策，只改变状态分组后重算信用，没有重新训练。每次细分都保留原有分组边界。</p><div class="table-scroll"><table><thead><tr><th>分组依据</th><th>组数</th><th>只有一次访问的组</th><th>step 项为零的决策</th><th>无效动作获正 step 项</th><th>无效动作获正合计</th></tr></thead><tbody>' + "".join(paired_rows) + '</tbody></table></div><p>分得更细后，能比较的样本变少了；无效动作的正局部权重减少，但正合计并没有随之减少。状态区分与共享样本数量需要一起考虑，不能只靠增加状态字段判断改进有效。</p><a href="' + esc(relative_url(paired_path, output)) + '">查看三种方法的完整重算记录</a></section>'
    page += '<section class="intro"><h2>哪些现象要分开看</h2><ul><li>v3 要求明确的单个 JSON 动作，比原生 DCC 解析严格。四组已核对的 766 次决策中，有 34 次正常说明前缀加唯一 JSON 被 v3 拒绝；这属于接口条件，不能归为算法缺陷。换协议也会改变同组轨迹与优势，静态解析对照不能替代重跑。</li><li>来源匹配区分大小写。仅忽略大小写时，GiGPO 与 episode-only 的同一道评测样例都会由失败变成功；这是评分敏感性，不是信用算法的差别。</li><li>本批多数轨迹一步就提交，观察到的主要是短路径和错误恢复，尚不能代表完整长程任务。没有做删除某一步再续跑的反事实实验，因此优势正负不等于动作的真实因果贡献。</li></ul></section>'
    page += '<section class="intro"><h2>怎样读步骤信用</h2><p>EP 是整条结果提供的信号；GiGPO 的 step、G2PO 的 NC/EC、GraphPO 的正确性/效率项另列。“实际合计”直接取已核对的训练记录，不把重复记录的分项相加。正数不等于这一步真的有用，负数也不是动作因果价值的证明。</p><p class="muted">“新增来源片段”按文本去重，不代表新增内容一定相关。相同错误反馈会标为重复反馈，不当成重复搜索。评测记录没有训练优势。</p></section>'
    page += '<div class="toolbar"><label>方法<select id="method"><option value="">全部方法</option>' + options + '</select></label><label>阶段<select id="phase"><option value="">训练与评测</option><option value="training">训练</option><option value="evaluation">评测</option></select></label><label>合同 / 任务关键词<input id="query" type="search" placeholder="例如 Change Of Control"></label><button id="open-visible" type="button">展开筛选结果</button><button id="close-visible" type="button">收起筛选结果</button><span class="count" id="visible-count"></span></div>'
    page += '<p class="muted">筛选只影响下方记录。轨迹和步骤均从 1 开始编号。</p><div id="rollouts">' + cards + '</div><p class="empty" id="empty" hidden>没有符合筛选条件的记录。</p>'
    evidence_path = output / "evidence/index.json"
    evidence_links = ""
    if evidence_path.exists():
        evidence_index = read_json(evidence_path)
        evidence_labels = {"gigpo_steps": "GiGPO 逐步核查", "episode_only": "Episode-only 核查",
                           "graphgpo_steps": "GraphGPO 逐步核查", "graphgpo_all_failure": "GraphGPO 全失败组",
                           "g2po_steps": "G2PO 逐步核查", "graphpo_representation": "GraphPO 状态与合并",
                           "graphpo_credit": "GraphPO 信用独立复算", "protocol_sensitivity": "动作解析差异",
                           "casefold_sensitivity": "来源大小写敏感性"}
        evidence_links = "<br>核查依据：" + " · ".join('<a href="' + esc(relative_url(evidence_path.parent / item["file"], output))
                        + '">' + esc(evidence_labels.get(key, key)) + '</a>' for key, item in evidence_index.items())
    page += '<footer>来源：各次运行的原始 JSONL、训练日志与记录一致性核验。页面生成时间：' + esc(timestamp) + '。<br><a href="summary.json">查看统计与文件校验 JSON</a>' + evidence_links + '</footer></main><script>' + JS + '</script></body></html>'
    atomic_text(output / "index.html", page)
    inspector = LinkInspector()
    inspector.feed(page)
    fail(len(inspector.ids) == len(set(inspector.ids)), "Generated HTML contains duplicate element IDs")
    fail(not inspector.external_dependencies, "Viewer must not load external dependencies")
    fail(len(re.findall(r'class="rollout"', page)) == len(lookup), "Rendered rollout card count differs from data")
    fail(esc('<script>alert("test")</script>') == '&lt;script&gt;alert(&quot;test&quot;)&lt;/script&gt;', "HTML escaping self-check failed")
    summary = {"title": "CUAD 基线轨迹对照", "generated_at": timestamp,
               "loaded_methods": methods, "not_yet_included": missing,
               "rollout_cards": len(lookup), "selected_cases": case_count, "runs": summaries,
               "validation": {"raw_audit_rollouts_and_actions_match": True,
                              "aggregate_counts_match": True, "unique_html_ids": len(inspector.ids),
                              "html_escape_self_check": True, "external_dependencies": 0,
                              "common_evaluation_task_set": common_evaluation}}
    atomic_text(output / "summary.json", dump(summary) + "\n")
    for link in inspector.links:
        parts = urlsplit(link)
        fail(not parts.scheme and not parts.netloc, f"Unexpected external link: {link}")
        if parts.path:
            fail((output / unquote(parts.path)).resolve().is_file(), f"Broken local file link: {link}")
        if parts.fragment:
            fail(unquote(parts.fragment) in inspector.ids, f"Broken local anchor: {link}")
    summary["validation"]["verified_links"] = len(inspector.links)
    summary["validation"]["javascript_controls"] = check_javascript(output / "index.html")
    summary["validation"]["html_sha256"] = digest(output / "index.html")
    summary["validation"]["html_bytes"] = (output / "index.html").stat().st_size
    atomic_text(output / "summary.json", dump(summary) + "\n")
    print(dump({"output": str(output / "index.html"), "methods": methods,
                "rollouts": len(lookup), "links_verified": len(inspector.links),
                "html_bytes": summary["validation"]["html_bytes"]}))


if __name__ == "__main__":
    main()
