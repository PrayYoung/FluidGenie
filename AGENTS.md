# GPU execution and monitoring

- Use the global `vast-run` backend by default, honoring `.vast/profile.toml` when present and the authorized pricing and cost safeguards. Let the runner own provisioning, SSH readiness, execution, bounded infrastructure retries, artifact retrieval, and cleanup.
- After submitting a GPU job, save its run ID, owner, artifact location, and current state, then return control to the user. Do not actively wait with repeated tool calls, poll progress, or repeatedly pull logs. The existing runner continues independently until its terminal result and cleanup.
- When the user asks for status, read the saved local runner metadata once. Analyze downloaded artifacts after completion; do not create duplicate jobs while an existing one may still be active.
- Distinguish infrastructure/workload failures from negative scientific outcomes. Do not rerun a negative result unchanged.
- Keep downloads small: reports/media, final usable checkpoints and resume state, and unique new inputs. Reuse verified local copies and exclude intermediate checkpoint duplicates. Keep credentials, personal information, and provider setup/configuration out of GitHub updates.
