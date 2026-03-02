from langgraph.store.base import BaseStore

REFERENCE_INDEX_KEY = "REF_INDEX"
REFERENCE_KEY_TEMPLATE = "REF_{index}"


def get_reference_key(store: BaseStore, user_id: str, value: str) -> str:
    """Create and return a new reference key for given value or return the existing reference key."""

    # Search for existing references.
    namespace = (user_id, "references")
    existing_reference = store.search(
        namespace,
        filter={"value": value},
        limit=1,
    )

    if len(existing_reference) == 0:
        # If no reference for the given value exists, create new reference.
        reference_index = 1

        # Get reference index.
        reference_index_item = store.get(namespace, REFERENCE_INDEX_KEY)

        # Setup reference index item, if it does not exist or is not an integer.
        if reference_index_item is None or not isinstance(
            reference_index_item.value["value"], int
        ):
            store.put(
                namespace=namespace,
                key=REFERENCE_INDEX_KEY,
                value={"value": reference_index},
            )
        else:
            # Get next reference index.
            reference_index = int(reference_index_item.value["value"]) + 1
            # Update reference index item with next index.
            store.put(
                namespace=namespace,
                key=REFERENCE_INDEX_KEY,
                value={"value": reference_index},
            )

        # Save new reference.
        reference_key = REFERENCE_KEY_TEMPLATE.format(index=reference_index)
        store.put(
            namespace=namespace,
            key=reference_key,
            value={"value": value},
        )

        return reference_key

    else:
        # Return the existing reference key for the given value.
        return existing_reference[0].key


def get_reference_value(store: BaseStore, user_id: str, key: str) -> str | None:
    """Return the value for the given reference key."""

    reference_item = store.get(
        namespace=(user_id, "references"),
        key=key,
    )

    return reference_item.value["value"] if reference_item is not None else None
