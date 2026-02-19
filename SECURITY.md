# Security Policy

## Reporting a Vulnerability

Please do not report security vulnerabilities in public issues.

Report privately to: `dev@dragonscale.ai`

Include:

- A clear description of the issue
- Affected versions/commits
- Reproduction steps or proof of concept
- Potential impact
- Suggested mitigation (if available)

## Response Process

- We will acknowledge receipt as soon as practical.
- We will investigate, reproduce, and assess impact.
- We will coordinate a fix and release.
- We will credit reporters if desired.

## Supported Versions

| Version | Supported |
| --- | --- |
| `0.1.x` | Yes |
| `< 0.1.0` | No |

## Scope Notes

This crate can integrate with local and remote providers. Some security risks may depend on:

- Provider API usage and credentials
- Host application deployment choices
- Model artifact sources and integrity controls

If uncertain whether something is security-relevant, report it anyway.
