# Optional ZIP text-member fallback

`--web-zip-member-fallback` is disabled by default. It is available in the native training CLI and shared native evaluation CLI, including Web evaluation. Set it identically for matched base and adapter runs. When enabled, online-Web training configuration and the Web evaluator's actual tool configuration record its enabled state and version. Default training configuration omits this option and preserves its original serialized fields. Python callers may pass `WebTools(web_zip_member_fallback=True)`.

The default ZIP reader atomically rejects an export if any supported member exceeds the decoded text character cap. With this option, only character-budget failures can leave that member unexposed while retaining other eligible complete members. The source-wide bounded `archive_members` directory marks each exclusion as `unexposed_text_limit`, with reason `member_text_character_limit` or `archive_text_character_limit` and `text_exposed=false`. Excluded members have no preview or text offsets. This is an incomplete view of an archive, not proof that excluded rows fail the task's conditions.

The cumulative extracted-text cap is checked before writing an entire member label and its literal text. No member or CSV row is truncated to fit that cap. A later smaller member may still fit. If no nonempty readable member remains, the entire read fails as before: no page, archive navigation or successful source is cached.

The ZIP download, member count, compressed/expanded byte, ratio, path/type, encryption, nested archive and CRC checks remain enforced. Corruption, unsupported encodings, malformed supported XML/JSON and other validation errors still fail atomically. Character fallback never retries a failed request or raises a budget. XLSX processing is unchanged.

Directory entries, previews and member labels are navigation. For this opt-in, an ordinary window containing only that metadata returns `source_text_exposed=false`; it cannot satisfy the episode's successful-fetch grounding requirement or register a newly read source. Fetch the same URL at a readable member's `content_start` to expose literal content. Cached row selection accepts only members marked readable and retains its existing exact-row provenance. Unsupported or skipped members cannot become selected rows or citations through directory metadata alone.

This option addresses an archive availability limitation. Synthetic ZIP fixture and security tests establish bounded reader behavior; no training improvement or benchmark score is claimed.
