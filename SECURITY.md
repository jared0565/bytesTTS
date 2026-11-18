# Security policy

## Supported versions

BytesTTS is in pre-release. Security fixes go to the `main` branch and the latest release.

| Version | Supported |
|---|---|
| 0.1.x / `main` | Yes |

## Reporting a vulnerability

Please report vulnerabilities **privately** through GitHub:
[Report a vulnerability](https://github.com/jared0565/bytesTTS/security/advisories/new) (the repository's
*Security* tab, *Report a vulnerability*). Do not open a public issue.

Include what you can:

- the affected version or commit;
- what an attacker controls (a voice package, a download server, a redirect, the input text, the environment);
- steps or a proof of concept;
- the impact you expect.

You will get an acknowledgement as soon as possible. Once a fix is ready, it is released with an advisory that
credits you, unless you prefer otherwise.

## Scope

In scope: anything that lets a voice package, a catalogue download, a redirect or input text do more than produce
audio. Examples: code execution, reading or writing files outside the voice directory, requests to local or private
network services, bypassing the SHA-256 pins, or running a program other than the configured espeak-ng.

The design and its known limits are described in the [Security model](docs/knowledge-base/security-model.md).
Issues listed there as known limits are not new vulnerabilities, although ideas for closing them are welcome.
Vulnerabilities in espeak-ng, ONNX Runtime or other dependencies should go to those projects; tell us too if
BytesTTS should work around them.
