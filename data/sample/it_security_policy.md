# IT Security Policy — Nordwind Mobility GmbH

*Fictional sample data for the RAG demo.*

## Passwords

Passwords for company accounts must be at least 14 characters long and must not be reused across services. Every employee receives a Bitwarden licence as the company password manager. Passwords are not rotated on a fixed schedule; they must be changed immediately if a compromise is suspected.

## Multi-factor authentication

Multi-factor authentication (MFA) is mandatory for all company accounts, including email, VPN and the HR portal. Administrators must use hardware security keys (YubiKey). SMS codes are not accepted as a second factor.

## Laptops and mobile devices

Full-disk encryption (BitLocker on Windows, FileVault on macOS) is mandatory on every company laptop. Screens must lock automatically after 5 minutes of inactivity. Operating-system security updates must be installed within 7 days of release. Lost or stolen devices must be reported immediately so they can be wiped remotely.

## Reporting security incidents

Suspected security incidents must be reported to the Information Security team within 1 hour of discovery, via security@nordwind.example or the 24/7 hotline. Do not try to investigate an incident yourself and do not delete evidence. Suspicious emails are reported with the "Report Phish" button in Outlook.

## Data classification

Company information is classified as Public, Internal, Confidential or Strictly Confidential. Confidential and Strictly Confidential data must not be uploaded to external AI tools or cloud services unless the tool has been approved by the Information Security team. Strictly Confidential data may only be shared in encrypted form.

## Network access

The company VPN must be used whenever working on public or hotel Wi-Fi. Private devices may connect to the guest network only, never to the internal office network.
