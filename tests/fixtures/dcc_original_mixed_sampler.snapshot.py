class MixedSampler:
    def __init__(self, records, cuad_tasks, tokenizer, runner, *, rank, world, seed, web_fraction,
                 negative_fraction, max_prompt, max_response, batch_tokens, cuad_records=(), cuad_tool_fraction=0.75,
                 cuad_tool_partial_stages=False, record_encoder=None, recorded_cuad_stages=False):
        encoder = encode_record if record_encoder is None else record_encoder
        self.rng = random.Random(seed + rank)
        self.tokenizer, self.runner = tokenizer, runner
        self.max_prompt, self.max_response, self.batch_tokens = max_prompt, max_response, batch_tokens
        self.skips, self.web, self.cuad = Counter(), defaultdict(list), defaultdict(lambda: defaultdict(list))
        self.cuad_tools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.cuad_missing_stages = []
        ids = sorted({record["task_id"] for record in records})
        assigned = set(ids[rank::world])
        for record in records:
            if record["task_id"] in assigned:
                example = encoder(record, tokenizer, max_prompt=max_prompt, max_response=max_response,
                                        batch_tokens=batch_tokens, skips=self.skips)
                if example is not None:
                    self.web[example.stage].append(example)
        for task in cuad_tasks:
            if task.benchmark != "cuad" or task.split != "train":
                raise ValueError("CUAD auxiliary accepts CUAD train tasks only")
            if not isinstance(task.gold, list) or any(not isinstance(span, str) or not span.strip() for span in task.gold):
                raise ValueError("CUAD gold must contain only nonempty spans")
        cuad_ids = sorted({task.id for task in cuad_tasks} | {record["task_id"] for record in cuad_records})
        assigned_cuad = set(cuad_ids[rank::world])
        for task in cuad_tasks:
            if task.id in assigned_cuad:
                self.cuad[bool(task.gold)][str(task.metadata.get("category", task.prompt))].append(task)
        for record in cuad_records:
            if record["task_id"] in assigned_cuad:
                example = encoder(record, tokenizer, max_prompt=max_prompt, max_response=max_response,
                                        batch_tokens=batch_tokens, skips=self.skips)
                if example is not None:
                    self.cuad_tools[record["metadata"]["positive"]][example.category][record["stage"]].append(example)
        for positive, categories in self.cuad_tools.items():
            for category, stages in categories.items():
                if set(stages) != set(CUAD_STAGES):
                    if not (recorded_cuad_stages and set(stages) <= set(CUAD_STAGES)
                            or cuad_tool_partial_stages and positive and set(stages) == {"search", "read"}):
                        raise ValueError(f"CUAD tool category {category!r} lost a full stage; raise budgets; skipped={dict(self.skips)}")
                    self.cuad_missing_stages.append({"positive": positive, "category": category,
                        "available_stages": [stage for stage in CUAD_STAGES if stage in stages],
                        "missing_stages": [stage for stage in CUAD_STAGES if stage not in stages]})
        if not self.web and not self.cuad and not self.cuad_tools:
            raise ValueError(f"No eligible local training examples fit the token budget; skipped={dict(self.skips)}")
        self.web_fraction = web_fraction if self.cuad or self.cuad_tools else 1.0
        if not self.web:
            if web_fraction > 0:
                raise ValueError(f"No local web teacher fits; increase budgets or reduce worker count; skipped={dict(self.skips)}")
            self.web_fraction = 0.0
        if self.web_fraction > 0 and set(self.web) != set(STAGES):
            raise ValueError(f"Every local web stage must retain an eligible example; skipped={dict(self.skips)}")
        self.negative_fraction = negative_fraction
        self.cuad_tool_fraction = cuad_tool_fraction
        self.stage_index, self.cuad_stage_index = rank, rank
        self.pending = None
        self.available = {stage: len(values) for stage, values in self.web.items()}

    def sample(self):
        if self.pending is not None:
            example, self.pending = self.pending, None
            return example
        if self.web and self.rng.random() < self.web_fraction:
            stage = STAGES[self.stage_index % len(STAGES)]
            self.stage_index += 1
            return self.rng.choice(self.web[stage])
        for _ in range(300):
            positive = self.rng.random() >= self.negative_fraction
            if positive not in self.cuad and positive not in self.cuad_tools:
                positive = not positive
            categories = self.cuad.get(positive, {})
            tool_categories = self.cuad_tools.get(positive, {})
            category = self.rng.choice(sorted(set(categories) | set(tool_categories)))
            if category in tool_categories and (category not in categories or self.rng.random() < self.cuad_tool_fraction):
                available_stages = [stage for stage in CUAD_STAGES if stage in tool_categories[category]]
                stage = available_stages[self.cuad_stage_index % len(available_stages)]
                self.cuad_stage_index += 1
                return self.rng.choice(tool_categories[category][stage])
            task = self.rng.choice(categories[category])
            example = supervised_example(task, self.runner, self.tokenizer, self.max_prompt, self.max_response)
            if example is None:
                self.skips["cuad_ungrounded_or_target_too_long"] += 1
                continue
            prompt, completion = example
            if len(prompt) + len(completion) > self.batch_tokens:
                self.skips["cuad_batch_budget"] += 1
                continue
            return Example(task.id, "cuad_positive" if positive else "cuad_negative", category, prompt, completion)
        raise RuntimeError("No grounded CUAD example fits after 300 draws; inspect budgets and retrieval")
