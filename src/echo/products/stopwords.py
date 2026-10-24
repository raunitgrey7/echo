"""English stopwords shared by the keyword extractor, the summariser and the search index.

Why: one list, one behaviour. The list is the usual function words plus conversational filler
that lectures and meetings are full of (okay, right, going, thing), which would otherwise
surface as keywords.
"""

from __future__ import annotations

FUNCTION_WORDS = (
    "a about above after again against all also am an and any are aren't as at be because been before being "
    "below between both but by can cannot could couldn't did didn't do does doesn't doing don't down during "
    "each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's "
    "hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's "
    "me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over "
    "own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs "
    "them themselves then there there's these they they'd they'll they're they've this those through to too "
    "under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where "
    "where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've "
    "your yours yourself yourselves"
)
FILLER_WORDS = (
    "okay ok right well yes yeah just really like thing things going go get got gets lot lots one two three "
    "today next last still even much many every another whole part say said says start starting "
    "look looking take takes taking make makes made come comes something anything everything nothing "
    "good great thanks thank sure item items agreed noted everyone round table "
    "will would shall may might must need needs needed want wants"
)

STOPWORDS: frozenset[str] = frozenset(FUNCTION_WORDS.split() + FILLER_WORDS.split())

COMMON_VERBS: frozenset[str] = frozenset(
    (
        "be have do say get make go know take see come think look want give use find tell ask work seem feel "
        "try leave call keep let begin help talk turn start show hear play run move like live believe hold "
        "bring happen write provide sit stand lose pay meet include continue set learn change lead understand "
        "watch follow stop create speak read allow add spend grow open walk win offer remember love consider "
        "appear buy wait serve die send expect build stay fall cut reach kill remain suggest raise pass sell "
        "require report decide pull describe explain measure relate enter travel limit store produce release "
        "assemble fix split absorb apply draw label discard compute return power miss implement sit cross "
        "clear bid respond bear lift form soak reach drive intensify speed estimate reshape fund exempt approve "
        "represent break declare storm symbolize state belong abolish execute end argue matter cover update "
        "schedule record rewrite share confirm post rerun grant remove flag point sync fire validate reject "
        "draft automate close review ship stream load log need want"
    ).split()
)


def is_verb_like(token: str) -> bool:
    """True when the token or its de-inflected stem is a common English verb."""
    if token in COMMON_VERBS:
        return True
    for suffix in ("ing", "ed", "es", "s", "d"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            stem = token[: -len(suffix)]
            if stem in COMMON_VERBS or stem + "e" in COMMON_VERBS:
                return True
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in COMMON_VERBS:
                return True
    return False
