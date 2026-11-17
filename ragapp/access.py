"""Access control: who may see which chunks.

* **Labels come from the documents.** At ingestion every chunk gets a classification
  (PUBLIC, INTERNAL, CONFIDENTIAL:HR, RESTRICTED:SECURITY, ...) read from the page it comes
  from ("Classification: CONFIDENTIAL:HR" or a title line "CONFIDENTIAL:HR — ..."). A page
  without a label inherits the cover page's label, else ``default_classification``
  (INTERNAL): deny by default, nothing becomes public by accident.
* **Identity comes from the credential, never from the text.** The API resolves the
  caller's API key to a user and role on the server ("As hr_demo, ..." in a question
  changes nothing). The role lists the labels it may see.
* **Filtering happens inside the search.** Qdrant only returns chunks whose label the
  caller may see, so forbidden text never reaches the reranker, the LLM, the cache,
  citations, logs or traces. The LLM is not the security boundary.
* **Not allowed looks like not found.** A forbidden question gets the normal "I don't
  know", so the answer does not reveal that a restricted document exists.

The identities file is a demo stand-in for a real identity provider: in production the
role would come from SSO/OIDC group claims, and the label-to-role mapping from policy.
"""

import hmac
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_LABEL = r"(PUBLIC|INTERNAL|(?:CONFIDENTIAL|RESTRICTED)(?::[A-Za-z][A-Za-z0-9_-]*)?)"
# "Classification: CONFIDENTIAL:HR | Record ..."  or a title line "RESTRICTED:Security — Secret canaries".
# Upper-case on purpose: the word "Restricted" in prose or a bare table cell "PUBLIC" is not a label.
LABEL_LINE = re.compile(
    rf"^[ \t]*(?:Classification[ \t]*:[ \t]*{_LABEL}|{_LABEL}[ \t]+[—–-][ \t]+\S)", re.MULTILINE
)
LEVELS = {"PUBLIC": 0, "INTERNAL": 1, "CONFIDENTIAL": 2, "RESTRICTED": 3}


def normalize_label(label: str) -> str:
    return label.strip().upper()


def find_label(text: str) -> str | None:
    """The first classification label stated in ``text``, normalised, or None."""
    match = LABEL_LINE.search(text)
    return normalize_label(match.group(1) or match.group(2)) if match else None


def level(label: str) -> int:
    return LEVELS.get(label.split(":", 1)[0], max(LEVELS.values()))  # unknown -> most restrictive


def is_sensitive(label: str) -> bool:
    """CONFIDENTIAL or RESTRICTED (any department)."""
    return level(label) >= LEVELS["CONFIDENTIAL"]


@dataclass(frozen=True)
class Identity:
    user: str
    role: str
    scopes: frozenset[str] | None = None  # labels this identity may see; None = everything
    can_ingest: bool = False

    def allows(self, label: str | None) -> bool:
        return self.scopes is None or (label is not None and label in self.scopes)

    def scope_list(self) -> list[str] | None:
        return None if self.scopes is None else sorted(self.scopes)

    def describe(self) -> dict:
        return {"user": self.user, "role": self.role, "scopes": self.scope_list() or ["*"]}


# Access control off, or the local operator (CLI): the index files are readable anyway.
UNRESTRICTED = Identity(user="local_operator", role="operator", scopes=None, can_ingest=True)


@dataclass
class AccessPolicy:
    users: dict[str, Identity]  # user name -> identity
    keys: dict[str, str]  # user name -> API key
    anonymous: Identity | None = None  # requests without a key; None = rejected
    roles: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, anonymous_role: str | None = None) -> "AccessPolicy":
        roles = {
            name: sorted({normalize_label(s) for s in spec.get("scopes", [])})
            for name, spec in data.get("roles", {}).items()
        }
        ingest_roles = {name for name, spec in data.get("roles", {}).items() if spec.get("can_ingest")}

        def identity(user: str, role: str) -> Identity:
            if role not in roles:
                raise ValueError(f"identities: user {user!r} has unknown role {role!r}")
            return Identity(user, role, frozenset(roles[role]), role in ingest_roles)

        users, keys = {}, {}
        for user, spec in data.get("users", {}).items():
            users[user] = identity(user, spec["role"])
            keys[user] = str(spec["key"])
        if len(set(keys.values())) != len(keys):
            raise ValueError("identities: two users share the same key")
        anonymous = identity("anonymous", anonymous_role) if anonymous_role else None
        return cls(users=users, keys=keys, anonymous=anonymous, roles=roles)

    @classmethod
    def from_file(cls, path: str | Path, anonymous_role: str | None = None) -> "AccessPolicy":
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(
                f"Access control is on but {file} does not exist. Create it (see config/identities.json) "
                "or set RAG_ACCESS_CONTROL=false."
            )
        return cls.from_dict(json.loads(file.read_text(encoding="utf-8")), anonymous_role)

    def resolve(self, api_key: str | None) -> Identity | None:
        """Identity for a credential: anonymous without one, None (reject) for an unknown one."""
        if not api_key:
            return self.anonymous
        for user, key in self.keys.items():
            if hmac.compare_digest(key.encode(), api_key.encode()):  # constant time
                return self.users[user]
        return None

    def user(self, name: str) -> Identity:
        if name not in self.users:
            raise KeyError(f"unknown user {name!r}; known: {', '.join(sorted(self.users))}")
        return self.users[name]
