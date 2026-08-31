from __future__ import annotations


def infinite_batches(dataloader):
    """Repeat epochs without ``itertools.cycle`` caching the full dataset."""
    while True:
        yield from dataloader
