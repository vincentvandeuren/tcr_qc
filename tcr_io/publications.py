"""Publication metadata from an identifier: CrossRef, PubMed Central, NCBI BioProject,
immuneACCESS.

Nothing here knows about a dataset — it maps a string to `StudyMetadata` records over HTTP,
and can be used, and tested, on its own. The operation that stores the result for a dataset
is `operations/publications.py`.
"""
from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, field, fields
from typing import Callable, List, Optional, Tuple

import requests

# A DOI *given as an identifier* is the whole string, parentheses and all — Lancet-style
# `10.1016/S0140-6736(20)30183-5` is a real DOI. A DOI *found in a page* has to stop at the
# quote or paren that surrounds it, which does truncate that style; there is no way to have
# both, and mangling every scraped DOI is worse than missing part of a rare one.
DOI = r'10\.\d{4,9}/[^\s"<>]+'
DOI_IN_PAGE = r'10\.\d{4,9}/[^\s"<>\')]+'

_MAX_LINKED_PUBMED = 5      # a BioProject can link dozens; the first few are the study

# A DOI a record points at, and the PMID it was reached through (None when there was none).
Linked = Tuple[str, Optional[str]]


@dataclass
class StudyMetadata:
    """One publication. `error` set means the fetch failed and the rest is empty.

    Failure is a record rather than an exception because the caller usually wants a whole
    batch: one unreachable identifier should not cost the other nine, and "why is this row
    blank" is a question the stored row itself can answer.
    """
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    source_type: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    journal: Optional[str] = None
    error: Optional[str] = None


# --- text ---------------------------------------------------------------------------------

def unescape(text: str) -> str:
    """HTML-unescape to a fixed point — CrossRef abstracts arrive doubly escaped."""
    while (once := html.unescape(text)) != text:
        text = once
    return text


def strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def clean(text: str) -> str:
    """Markup out, entities resolved, whitespace collapsed."""
    return re.sub(r"\s+", " ", unescape(strip_tags(text))).strip()


def _find(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    """First capture group, cleaned — the shape of nearly every extraction below."""
    m = re.search(pattern, text, flags)
    return clean(m.group(1)) if m else None


def _year(text: str, *patterns: str) -> Optional[int]:
    """First four-digit year found by any of `patterns`, tried in order.

    One helper for three different date shapes — a `<pub-date>` element, a `projectYear`
    assignment, an ISO registration date — because what all three want is the same: the
    first pattern that yields four digits, ignoring the ones that yield junk.
    """
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m and (digits := re.search(r"\d{4}", m.group(1))):
            return int(digits.group())
    return None


def _dois_in(page: str) -> List[str]:
    """Every DOI on a page, de-duplicated, in order of appearance."""
    return list(dict.fromkeys(d.rstrip("/.") for d in re.findall(DOI_IN_PAGE, page)))


# --- http ---------------------------------------------------------------------------------

def _get(url: str, *, headers: Optional[dict] = None, timeout: int = 15,
         retries: int = 3) -> requests.Response:
    """GET, backing off through NCBI's 429. Any other error status raises."""
    for attempt in range(retries):
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        time.sleep(1.5 * (attempt + 1))
    resp.raise_for_status()
    return resp


def _eutils(endpoint: str, **params) -> dict:
    """One NCBI E-utilities JSON call. The four call sites differ only in endpoint and query."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return _get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}.fcgi"
                f"?{query}&retmode=json").json()


# --- fetchers -----------------------------------------------------------------------------
# Each returns (the record this identifier names, DOIs that record points at). They RAISE;
# `fetch_publication` turns a failure into an error record, once, for all of them.

def _crossref(doi: str) -> Tuple[StudyMetadata, List[Linked]]:
    data = _get(f"https://api.crossref.org/works/{doi}",
                headers={"Accept": "application/json"}).json()["message"]

    def first(key: str) -> Optional[str]:
        values = data.get(key) or []
        return clean(values[0]) if values else None

    year = None
    for key in ("published-print", "published-online", "issued"):
        parts = data.get(key, {}).get("date-parts") or [[]]
        if parts[0] and parts[0][0]:
            year = int(parts[0][0])
            break

    authors = [name for a in data.get("author", [])
               if (name := f"{a.get('given', '')} {a.get('family', '')}".strip())]
    return StudyMetadata(
        source_type="doi", doi=doi, title=first("title"),
        abstract=first("abstract"), authors=authors, year=year,
        journal=first("container-title"),
    ), []


def _pmc(pmc_id: str) -> Tuple[StudyMetadata, List[Linked]]:
    xml = _get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
               f"?db=pmc&id={pmc_id}&rettype=xml&retmode=xml").text

    authors = []
    for block in re.findall(r'<contrib[^>]*contrib-type="author"[^>]*>(.*?)</contrib>',
                            xml, re.DOTALL):
        surname = _find(r"<surname[^>]*>(.*?)</surname>", block)
        if surname:
            given = _find(r"<given-names[^>]*>(.*?)</given-names>", block) or ""
            authors.append(f"{given} {surname}".strip())

    doi = _find(r'<article-id[^>]*pub-id-type="doi"[^>]*>(.*?)</article-id>', xml)
    pmid = _find(r'<article-id[^>]*pub-id-type="pmid"[^>]*>(.*?)</article-id>', xml)
    record = StudyMetadata(
        source_type="pmc", doi=doi, pmid=pmid, authors=authors,
        title=_find(r"<article-title[^>]*>(.*?)</article-title>", xml, re.DOTALL),
        abstract=_find(r"<abstract[^>]*>(.*?)</abstract>", xml, re.DOTALL),
        journal=_find(r"<journal-title[^>]*>(.*?)</journal-title>", xml),
        year=_year(xml,
                   r'<pub-date[^>]*pub-type="ppub"[^>]*>.*?<year>(.*?)</year>',
                   r'<pub-date[^>]*pub-type="epub"[^>]*>.*?<year>(.*?)</year>',
                   r'<pub-date[^>]*pub-type="collection"[^>]*>.*?<year>(.*?)</year>',
                   r"<pub-date[^>]*>.*?<year>(.*?)</year>"),
    )
    return record, [(doi, pmid)] if doi else []


def _adaptive(url: str) -> Tuple[StudyMetadata, List[Linked]]:
    page = _get(url).text

    title = _find(r"<title[^>]*>(.*?)</title>", page, re.DOTALL)
    if title:   # the tab title is wrapped in site boilerplate on both sides
        title = re.sub(r"immuneACCESS\s*Data\s*\|\s*|\s*\|\s*immuneACCESS.*", "",
                       title).strip() or None

    # The `<meta description>` is often a site blurb rather than the study's; when it looks
    # too short to be an abstract, prefer the page's own projectAbstract field.
    abstract = _find(r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']',
                     page, re.DOTALL)
    if not abstract or len(abstract) < 50:
        abstract = _find(r'["\']?projectAbstract["\']?\s*[:=]\s*["\'](.+?)["\']',
                         page, re.DOTALL) or abstract

    investigators = _find(r'["\']?projectInvestigator["\']?\s*[:=]\s*["\'](.+?)["\']', page)
    authors = [name for part in re.split(r";|\band\b", investigators or "")
               if (name := part.strip().strip(","))]

    record = StudyMetadata(
        source_type="adaptive", title=title, abstract=abstract,
        authors=authors, year=_year(page, r'["\']?projectYear["\']?\s*[:=]\s*["\']?(\d{4})'),
    )
    return record, [(doi, None) for doi in _dois_in(page)]


def _bioproject(accession: str) -> Tuple[StudyMetadata, List[Linked]]:
    ids = _eutils("esearch", db="bioproject", term=accession) \
        .get("esearchresult", {}).get("idlist", [])
    if not ids:
        raise LookupError(f"BioProject {accession} not found")
    uid = ids[0]

    summary = _eutils("esummary", db="bioproject", id=uid).get("result", {}).get(str(uid), {})
    abstract = summary.get("project_description")
    org = next((summary[k] for k in ("project_data_provider", "registration_organization",
                                     "submitter_organization") if summary.get(k)), None)
    record = StudyMetadata(
        source_type="bioproject",
        title=summary.get("project_name") or summary.get("project_title"),
        abstract=clean(abstract) if abstract else None,
        authors=[org] if org else [],
        year=_year(summary.get("registration_date", ""), r"(\d{4})"),
    )

    time.sleep(0.4)     # NCBI asks for no more than three requests a second
    links = _eutils("elink", dbfrom="bioproject", db="pubmed", id=uid)
    pmids = [str(p) for linkset in links.get("linksets", [])
             for db in linkset.get("linksetdbs", []) if db.get("dbto") == "pubmed"
             for p in db.get("links", [])]

    linked = []
    for pmid in pmids[:_MAX_LINKED_PUBMED]:
        time.sleep(0.35)
        if doi := _pmid_to_doi(pmid):
            linked.append((doi, pmid))
    return record, linked


def _pmid_to_doi(pmid: str) -> Optional[str]:
    """The DOI a PubMed entry carries, or None. Swallows its errors on purpose: this runs
    inside the BioProject enrichment loop, where one dead link must not cost the record."""
    try:
        entry = _eutils("esummary", db="pubmed", id=pmid).get("result", {}).get(pmid, {})
        return next((a.get("value") for a in entry.get("articleids", [])
                     if a.get("idtype") == "doi"), None)
    except Exception:
        return None


# --- dispatch -----------------------------------------------------------------------------

@dataclass(frozen=True)
class _Source:
    """A recognizable kind of identifier: how to spot it, and how to fetch it.

    The pattern's first group IS the extracted id, so recognizing and extracting are one
    regex rather than the `check_identifier`/`extract_id` pair (times four subclasses of an
    ABC) this used to be.
    """
    type: str
    pattern: str
    fetch: Callable[[str], Tuple[StudyMetadata, List[Linked]]]


# Order matters: an immuneACCESS page and a PMC article both contain DOIs, so the DOI
# pattern has to come last or it would claim them.
_SOURCES = (
    _Source("adaptive",   r"(\S*adaptivebiotech\.com/pub/\S*)", _adaptive),
    _Source("pmc",        r"(PMC\d+)",                          _pmc),
    _Source("bioproject", r"^(PRJNA\d+)$",                      _bioproject),
    _Source("doi",        f"({DOI})",                           _crossref),
)


def _fill(record: StudyMetadata, extra: StudyMetadata) -> None:
    """Fill `record`'s empty fields from `extra`, in place. Never overwrites a value."""
    for f in fields(record):
        if not getattr(record, f.name) and getattr(extra, f.name):
            setattr(record, f.name, getattr(extra, f.name))


def fetch_publication(identifier: str) -> StudyMetadata:
    """One publication's metadata, from whatever kind of identifier names it.

    A record often has to be assembled from more than one source — an immuneACCESS page
    carries no abstract but prints the DOI of the paper that does, a BioProject links out to
    its publications. Those extra sources FILL GAPS in the primary record rather than
    becoming records of their own: what the caller asked about is one publication, so what
    comes back is one publication.

    Never raises. An unreachable host, an unparseable page and an unrecognized string all
    come back as a record with `error` set and the rest empty.
    """
    identifier = identifier.strip()
    for source in _SOURCES:
        m = re.search(source.pattern, identifier, re.IGNORECASE)
        if not m:
            continue
        try:
            record, linked = source.fetch(m.group(1).rstrip("/."))
        except Exception as e:
            return StudyMetadata(source_type=source.type, error=f"{type(e).__name__}: {e}")
        for doi, pmid in linked:
            try:
                extra = _crossref(doi)[0]
            except Exception:
                continue        # a pointed-at DOI that will not resolve is not this fetch failing
            extra.pmid = pmid
            _fill(record, extra)
        return record
    return StudyMetadata(source_type="unknown", error="unrecognized identifier")
