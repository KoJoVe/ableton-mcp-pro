# Security policy

## Supported version

Only the latest commit on this fork's `main` branch is supported. Install from
an immutable commit and use the checked-in lockfiles.

## Safe defaults

The MCP server and the Ableton Remote Script both start in read-only mode.
Absolute audio-file paths are redacted by default. Destructive operations need
both an administrator opt-in and explicit confirmation on every call.

The TCP listener remains unauthenticated by project-owner decision. It binds
only to `127.0.0.1`, but any local process running as the same user can attempt
to connect. Disable the Ableton control surface whenever it is not in use.

## Reporting a vulnerability

Do not open a public issue for an undisclosed vulnerability. Use GitHub's
private vulnerability reporting feature on this fork. Include affected commit,
reproduction steps, impact, and any proposed mitigation.

## Work-computer policy

Installation on a managed device requires employer approval. Do not expose
confidential Live sets to an AI provider unless company policy explicitly
permits it. Keep file-path redaction enabled, use a dedicated virtual
environment, and do not enable custom model sources.

