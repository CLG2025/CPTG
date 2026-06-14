# Security Policy

## Supported Versions

Security updates are provided for the latest publicly released version of this project.

| Version                                    | Supported |
| ------------------------------------------ | --------- |
| Latest release                             | Yes       |
| Earlier releases                           | No        |
| Development or modified third-party builds | No        |

Users should reproduce any reported issue using the latest official release before submitting a security report.

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub Issues, Discussions, pull requests, or other public channels.

Use GitHub's **private vulnerability reporting** feature for this repository:

1. Open the repository's **Security** tab.
2. Select **Advisories**.
3. Select **Report a vulnerability**.
4. Provide the requested technical details.

If private vulnerability reporting is not available, contact the repository owner privately through the contact method listed on the repository profile.

Please include:

* A clear description of the vulnerability.
* The affected version, release, or commit.
* The operating system and environment.
* Steps required to reproduce the issue.
* The potential security impact.
* Relevant logs, screenshots, or proof-of-concept material.
* Any suggested mitigation, if known.

Do not include confidential information, credentials, access tokens, private data, or destructive proof-of-concept code unless specifically requested through a secure channel.

## Responsible Disclosure

Please allow a reasonable period for investigation and remediation before publicly disclosing a vulnerability.

The project maintainer will make a good-faith effort to:

* Acknowledge a complete report.
* Review and reproduce the reported issue.
* Determine its severity and scope.
* Develop and test a correction when appropriate.
* Coordinate public disclosure after a correction or mitigation is available.

Response and remediation times depend on the severity, complexity, reproducibility, and maintainer availability. No fixed resolution deadline is guaranteed.

## Security Scope

Security reports may include issues involving:

* Unauthorized file access or modification.
* Arbitrary code execution.
* Command or path injection.
* Unsafe handling of user-supplied files or paths.
* Exposure of sensitive local information.
* Insecure network behavior.
* Dependency vulnerabilities that directly affect this project.
* Release-package tampering or integrity concerns.

The following are generally outside the security-reporting scope:

* General bugs without a security impact.
* Feature requests.
* Issues caused only by unsupported or modified builds.
* Vulnerabilities in third-party software that do not affect this project.
* Social-engineering reports without a project-specific technical vulnerability.
* Automated scanner results without reproducible evidence or demonstrated impact.

Non-security bugs should be reported through the repository's normal GitHub Issues process.

## Local Application Security

This project is intended to operate locally unless the repository documentation explicitly states otherwise.

Users should:

* Download releases only from the official repository.
* Verify release checksums when provided.
* Keep the operating system and required dependencies updated.
* Avoid running the software with administrator privileges unless necessary.
* Review local firewall prompts before allowing network access.
* Avoid opening untrusted or unexpectedly modified input files.
* Do not expose a local application server directly to the public internet.

## Safe Harbor

Good-faith security research is welcomed when it:

* Avoids privacy violations, data destruction, service disruption, and unauthorized access.
* Uses only the minimum testing necessary to demonstrate the issue.
* Does not exploit the vulnerability beyond what is required for verification.
* Keeps vulnerability details confidential during the remediation process.
* Complies with applicable laws and regulations.

The maintainer will not pursue action against researchers who follow this policy and act in good faith.

## Security Updates

Confirmed security corrections may be documented through GitHub Security Advisories, release notes, repository notices, or updated releases.

Users are responsible for keeping their installation current and reviewing security-related release information.
::: 
