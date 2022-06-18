from owlready2 import DataPropertyClass


def find_subjects(subject, subjects, predicates, depth=0):
    subject_properties = predicates.intersection(subject.get_properties())
    results = []
    for property in subject_properties:
        results.append(
            find_predicates(subject, property, subjects, predicates, depth + 1)
        )
    return results


def find_predicates(subject, predicate, subjects, predicates, depth=0):
    if isinstance(predicate, DataPropertyClass):
        return {"value": getattr(subject, predicate.name), "depth": depth}

    results = []
    property_subjects = subjects.intersection(set(predicate.get_range()))
    for subject in property_subjects:
        results.append(find_subjects(subject, subjects, predicates, depth + 1))

    return results
