"""Conservative tag evidence. Attributes never substitute for object nouns."""
import re


ATTRIBUTES = set('a an the one two three single large small big little white black red blue green yellow pink gray grey brown silver golden rectangular round long short own left right of on with and'.split())
ALIASES = ({'cup', 'teacup', 'mug'}, {'rock', 'rocks'}, {'detonator', 'detonators'},
           {'gun', 'guns'}, {'sword', 'swords'}, {'box', 'boxes'})


def words(value):
    return set(re.findall(r'[a-z0-9]+', str(value).lower()))


def object_evidence(objects, general):
    """Match each object's *name*, not its appearance or a substring of a tag.

    This only establishes a noun signal, never count/shape/contact correctness.
    Uninterpretable names remain unknown instead of matching their attributes.
    """
    signals = {str(tag): float(score) for tag, score in general.items() if float(score) >= .35}
    confirmed, missing, unsupported = {}, [], []
    for obj in objects:
        name = str(obj.get('name', '')).strip()
        tokens = words(name) - ATTRIBUTES
        if not tokens:
            unsupported.append(name)
            continue
        noun_groups = [aliases for aliases in ALIASES if tokens & aliases]
        if noun_groups:
            # Once a known noun is present, surrounding adjectives such as
            # 'coastal' must not independently confirm a rock.
            hits = {tag: score for tag, score in signals.items()
                    if any(words(tag) & aliases for aliases in noun_groups)}
        else:
            # Unknown names need their whole remaining phrase, not one token.
            hits = {tag: score for tag, score in signals.items() if tokens <= words(tag)}
        if hits:
            confirmed[name] = hits
        else:
            missing.append(name)
    return {'confirmed_by_object': confirmed, 'missing_noun_signals': missing,
            'unsupported_names': unsupported, 'evidence_type': 'noun_tag_signal',
            'unverified': ['count', 'attributes', 'shape', 'spatial_and_contact_relations']}
