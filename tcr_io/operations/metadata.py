from .base import BaseOperation, OperationResults

import re
import html
import time
import requests
from dataclasses import dataclass, field
from typing import Optional, List
from abc import ABC, abstractmethod
from dataclasses import asdict

# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class StudyMetadata:
    identifier: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    source_type: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    journal: Optional[str] = None
    investigator: Optional[str] = None
    sra_accession: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ClassifiedIdentifier:
    raw: str
    extracted_id: str
    source_type: str


# ─── Utilities ────────────────────────────────────────────────────────────────

def deep_unescape(text: str) -> str:
    if not text:
        return text
    prev = None
    while prev != text:
        prev = text
        text = html.unescape(text)
    return text


def strip_tags(text: str) -> str:
    if not text:
        return text
    return re.sub(r'<[^>]+>', '', text)


def clean_text(text: str) -> str:
    if not text:
        return text
    text = strip_tags(text)
    text = deep_unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def ncbi_get(url: str, timeout: int = 15, max_retries: int = 3) -> requests.Response:
    """GET with retry + backoff for NCBI rate limits."""
    for attempt in range(max_retries):
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 429:
            wait = 1.5 * (attempt + 1)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()  # raise last error
    return resp


# ─── Identifier Classification ────────────────────────────────────────────────

class IdentifierClassifier(ABC):
    @abstractmethod
    def check_identifier(self, identifier: str) -> bool:
        pass

    @abstractmethod
    def extract_id(self, identifier: str) -> str:
        pass

    @property
    @abstractmethod
    def source_type(self) -> str:
        pass


class AdaptiveIdentifier(IdentifierClassifier):
    source_type = "adaptive"

    def check_identifier(self, identifier: str) -> bool:
        return 'adaptivebiotech.com/pub/' in identifier

    def extract_id(self, identifier: str) -> str:
        return identifier.strip().rstrip('/')


class PMCIdentifier(IdentifierClassifier):
    source_type = "pmc"

    def check_identifier(self, identifier: str) -> bool:
        return bool(re.search(r'PMC\d+', identifier, re.IGNORECASE))

    def extract_id(self, identifier: str) -> str:
        m = re.search(r'(PMC\d+)', identifier, re.IGNORECASE)
        return m.group(1) if m else identifier


class BioProjectIdentifier(IdentifierClassifier):
    source_type = "bioproject"

    def check_identifier(self, identifier: str) -> bool:
        return bool(re.match(r'^PRJNA\d+$', identifier.strip(), re.IGNORECASE))

    def extract_id(self, identifier: str) -> str:
        return identifier.strip()


class DOIIdentifier(IdentifierClassifier):
    source_type = "doi"

    def check_identifier(self, identifier: str) -> bool:
        return bool(re.search(r'10\.\d{4,9}/[^\s]+', identifier))

    def extract_id(self, identifier: str) -> str:
        m = re.search(r'(10\.\d{4,9}/[^\s"<>]+)', identifier)
        return m.group(1).rstrip('/.') if m else identifier


CLASSIFIERS: List[IdentifierClassifier] = [
    AdaptiveIdentifier(),
    PMCIdentifier(),
    BioProjectIdentifier(),
    DOIIdentifier(),
]


def classify_id(identifier: str) -> ClassifiedIdentifier:
    identifier = identifier.strip()
    for clf in CLASSIFIERS:
        if clf.check_identifier(identifier):
            return ClassifiedIdentifier(
                raw=identifier,
                extracted_id=clf.extract_id(identifier),
                source_type=clf.source_type,
            )
    return ClassifiedIdentifier(raw=identifier, extracted_id=identifier, source_type="unknown")


# ─── Fetchers ─────────────────────────────────────────────────────────────────

def fetch_doi_metadata(doi: str) -> StudyMetadata:
    url = f"https://api.crossref.org/works/{doi}"
    try:
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()["message"]

        title_list = data.get("title", [])
        title = clean_text(title_list[0]) if title_list else None

        raw_abstract = data.get("abstract", "")
        abstract = clean_text(raw_abstract) if raw_abstract else None

        authors = []
        for a in data.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)

        year = None
        for date_key in ("published-print", "published-online", "issued"):
            dp = data.get(date_key, {}).get("date-parts", [[]])
            if dp and dp[0] and dp[0][0]:
                year = int(dp[0][0])
                break

        journal_list = data.get("container-title", [])
        journal = clean_text(journal_list[0]) if journal_list else None

        return StudyMetadata(
            identifier=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            source_type="doi",
            doi=doi,
            journal=journal,
        )
    except Exception as e:
        return StudyMetadata(identifier=doi, source_type="doi", doi=doi, error=str(e))


def fetch_pmc_metadata(pmc_id: str) -> List[StudyMetadata]:
    efetch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={pmc_id}&rettype=xml&retmode=xml"
    )
    results = []

    try:
        resp = ncbi_get(efetch_url)
        xml = resp.text

        # Title
        title_m = re.search(r'<article-title[^>]*>(.*?)</article-title>', xml, re.DOTALL)
        title = clean_text(title_m.group(1)) if title_m else None

        # Abstract
        abstract_m = re.search(r'<abstract[^>]*>(.*?)</abstract>', xml, re.DOTALL)
        abstract = clean_text(abstract_m.group(1)) if abstract_m else None

        # Authors — handle attributes on given-names/surname tags
        authors = []
        for cm in re.finditer(
            r'<contrib[^>]*contrib-type="author"[^>]*>(.*?)</contrib>', xml, re.DOTALL
        ):
            block = cm.group(1)
            surname_m = re.search(r'<surname[^>]*>(.*?)</surname>', block)
            given_m = re.search(r'<given-names[^>]*>(.*?)</given-names>', block)
            if surname_m:
                surname = strip_tags(surname_m.group(1)).strip()
                given = strip_tags(given_m.group(1)).strip() if given_m else ""
                name = f"{given} {surname}".strip()
                if name:
                    authors.append(name)

        # Year
        year = None
        for pattern in [
            r'<pub-date[^>]*pub-type="ppub"[^>]*>.*?<year>(.*?)</year>',
            r'<pub-date[^>]*pub-type="epub"[^>]*>.*?<year>(.*?)</year>',
            r'<pub-date[^>]*pub-type="collection"[^>]*>.*?<year>(.*?)</year>',
            r'<pub-date[^>]*>.*?<year>(.*?)</year>',
        ]:
            year_m = re.search(pattern, xml, re.DOTALL)
            if year_m:
                try:
                    year = int(year_m.group(1))
                    break
                except ValueError:
                    pass

        # DOI
        doi_m = re.search(r'<article-id[^>]*pub-id-type="doi"[^>]*>(.*?)</article-id>', xml)
        doi = doi_m.group(1).strip() if doi_m else None

        # PMID
        pmid_m = re.search(r'<article-id[^>]*pub-id-type="pmid"[^>]*>(.*?)</article-id>', xml)
        pmid = pmid_m.group(1).strip() if pmid_m else None

        # Journal
        journal_m = re.search(r'<journal-title[^>]*>(.*?)</journal-title>', xml)
        journal = clean_text(journal_m.group(1)) if journal_m else None

        pmc_meta = StudyMetadata(
            identifier=pmc_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            source_type="pmc",
            doi=doi,
            pmid=pmid,
            journal=journal,
        )
        results.append(pmc_meta)

        # Also fetch DOI metadata if available
        if doi:
            doi_meta = fetch_doi_metadata(doi)
            if not doi_meta.error:
                doi_meta.pmid = pmid
                results.append(doi_meta)

    except Exception as e:
        results.append(StudyMetadata(identifier=pmc_id, source_type="pmc", error=str(e)))

    return results


def fetch_adaptive_metadata(url: str) -> List[StudyMetadata]:
    results = []

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        page = resp.text

        # Title
        title = None
        title_m = re.search(r'<title[^>]*>(.*?)</title>', page, re.DOTALL)
        if title_m:
            raw_title = strip_tags(title_m.group(1)).strip()
            raw_title = re.sub(r'immuneACCESS\s*Data\s*\|\s*', '', raw_title)
            raw_title = re.sub(r'\s*\|\s*immuneACCESS.*', '', raw_title)
            title = deep_unescape(raw_title).strip() or None

        # Abstract
        abstract = None
        meta_m = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']', page, re.DOTALL
        )
        if meta_m:
            abstract = clean_text(meta_m.group(1))

        if not abstract or len(abstract) < 50:
            abs_m = re.search(r'["\']?projectAbstract["\']?\s*[:=]\s*["\'](.+?)["\']', page, re.DOTALL)
            if abs_m:
                abstract = clean_text(abs_m.group(1))

        # Authors
        authors = []
        inv_m = re.search(r'["\']?projectInvestigator["\']?\s*[:=]\s*["\'](.+?)["\']', page)
        if inv_m:
            raw_authors = deep_unescape(inv_m.group(1))
            parts = re.split(r'[;]|\band\b', raw_authors)
            for part in parts:
                part = part.strip().strip(',')
                if part:
                    authors.append(part)

        # Year
        year = None
        year_m = re.search(r'["\']?projectYear["\']?\s*[:=]\s*["\']?(\d{4})["\']?', page)
        if year_m:
            year = int(year_m.group(1))

        # DOIs on page
        dois_on_page = re.findall(r'(10\.\d{4,9}/[^\s"<>\')]+)', page)
        seen = set()
        unique_dois = []
        for d in dois_on_page:
            d_clean = d.rstrip('/.')
            if d_clean not in seen:
                seen.add(d_clean)
                unique_dois.append(d_clean)

        adaptive_meta = StudyMetadata(
            identifier=url,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            source_type="adaptive",
        )
        results.append(adaptive_meta)

        for doi in unique_dois:
            doi_meta = fetch_doi_metadata(doi)
            if not doi_meta.error:
                results.append(doi_meta)

    except Exception as e:
        results.append(StudyMetadata(identifier=url, source_type="adaptive", error=str(e)))

    return results


def fetch_bioproject_metadata(accession: str) -> List[StudyMetadata]:
    results = []

    try:
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=bioproject&term={accession}&retmode=json"
        )
        resp = ncbi_get(search_url)
        id_list = resp.json().get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return [StudyMetadata(identifier=accession, source_type="bioproject", error="Not found")]

        uid = id_list[0]

        summary_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=bioproject&id={uid}&retmode=json"
        )
        resp = ncbi_get(summary_url)
        summary = resp.json().get("result", {}).get(str(uid), {})

        title = summary.get("project_name") or summary.get("project_title")
        abstract = summary.get("project_description")
        if abstract:
            abstract = clean_text(abstract)

        authors = []
        for key in ("project_data_provider", "registration_organization", "submitter_organization"):
            org = summary.get(key)
            if org:
                authors = [org]
                break

        year = None
        date_str = summary.get("registration_date", "")
        year_m = re.search(r'(\d{4})', date_str)
        if year_m:
            year = int(year_m.group(1))

        bp_meta = StudyMetadata(
            identifier=accession,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            source_type="bioproject",
        )
        results.append(bp_meta)

        # Linked PubMed → DOI
        time.sleep(0.4)  # rate limit buffer
        link_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
            f"?dbfrom=bioproject&db=pubmed&id={uid}&retmode=json"
        )
        resp = ncbi_get(link_url)
        link_data = resp.json()

        pmids = []
        for linkset in link_data.get("linksets", []):
            for linksetdb in linkset.get("linksetdbs", []):
                if linksetdb.get("dbto") == "pubmed":
                    pmids.extend(linksetdb.get("links", []))

        for pmid in pmids[:5]:
            time.sleep(0.35)
            doi = _pmid_to_doi(str(pmid))
            if doi:
                doi_meta = fetch_doi_metadata(doi)
                if not doi_meta.error:
                    doi_meta.pmid = str(pmid)
                    results.append(doi_meta)

    except Exception as e:
        results.append(StudyMetadata(identifier=accession, source_type="bioproject", error=str(e)))

    return results


def _pmid_to_doi(pmid: str) -> Optional[str]:
    try:
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&id={pmid}&retmode=json"
        )
        resp = ncbi_get(url)
        data = resp.json().get("result", {}).get(pmid, {})
        for aid in data.get("articleids", []):
            if aid.get("idtype") == "doi":
                return aid.get("value")
    except Exception:
        pass
    return None


# ─── Main Dispatch ────────────────────────────────────────────────────────────

def fetch_metadata(classified: ClassifiedIdentifier) -> List[StudyMetadata]:
    dispatch = {
        "doi": lambda: [fetch_doi_metadata(classified.extracted_id)],
        "pmc": lambda: fetch_pmc_metadata(classified.extracted_id),
        "adaptive": lambda: fetch_adaptive_metadata(classified.extracted_id),
        "bioproject": lambda: fetch_bioproject_metadata(classified.extracted_id),
    }

    handler = dispatch.get(classified.source_type)
    if handler:
        return handler()

    return [StudyMetadata(
        identifier=classified.raw,
        source_type="unknown",
        error="Unrecognized identifier type",
    )]



@dataclass
class MetadataFetcher(BaseOperation):
    name = "metadata_fetcher"
    version = "0.1"
    description = "Fetches study metadata (title, abstract, authors, year, etc.) based on provided identifiers"

    def _run(self, ds, locus: Optional[str] = None) -> OperationResults:
        result_ndjson = []

        publication_ids = ds.publication_meta["publication_id"]
        for pub_id in publication_ids:
            pub_cl = classify_id(pub_id)
            if pub_cl.source_type != "unknown":
                pub_meta = fetch_metadata(pub_cl)
                for r in pub_meta:
                    r = asdict(r)
                    r["original_id"] = pub_id
                    result_ndjson.append(r)

        return OperationResults(
            outputs={"publication_metadata": result_ndjson}
        )
        