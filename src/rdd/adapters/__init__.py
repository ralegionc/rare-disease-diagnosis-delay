"""Ingest adapters: real claims sources -> the canonical schema.

Each adapter is responsible for exactly one thing -- producing a valid
:class:`rdd.schema.ClaimsDataset` -- and for being explicit about what it had
to invent. Every real source is missing something the canonical schema wants
(SynPUF has no rendering-provider specialty; many extracts have no death date),
and an adapter that quietly fills those with plausible defaults would push a
data problem into the results where nobody can see it. So each adapter returns
a ``notes`` list naming its substitutions, and the CLI prints it.

Available:

``generic``
    a two-CSV contract you can map any licensed extract onto;
``synpuf``
    CMS Data Entrepreneurs' Synthetic Public Use File (2008-2010);
``omop``
    OMOP CDM v5.4, via SQL against any supported backend.
"""

from .generic import load_generic  # noqa: F401
from .omop import OMOP_QUERIES, load_omop  # noqa: F401
from .synpuf import load_synpuf  # noqa: F401

__all__ = ["load_generic", "load_synpuf", "load_omop", "OMOP_QUERIES"]
