"""Text normalisation for the supplied PDFs.

pypdf renders these files with two artifacts that break naive parsing: every
space is doubled, and wrapped lines are emitted one word per line separated by
`\\n \\n`. Left alone, a clause reads as

    charge\\n \\nINR\\n \\n250\\n \\nunless

which no clause segmenter or number extractor can work with. Normalisation is
therefore a real step, not tidying.
"""

from __future__ import annotations

import pytest

from src.knowledge.pdftext import extract_text, normalise
from src.knowledge.sources import SOURCE_FILES


class TestArtifactRepair:
    def test_rejoins_word_per_line_runs(self):
        raw = "charge\n \nINR\n \n250\n \nunless\n \na\n \ncustomer"
        assert normalise(raw) == "charge INR 250 unless a customer"

    def test_collapses_doubled_spaces(self):
        assert normalise("ParcelPilot  Support  Policy  v3") == "ParcelPilot Support Policy v3"

    def test_normalises_non_breaking_spaces(self):
        assert normalise("INR\u00a0250") == "INR 250"

    def test_drops_soft_hyphens(self):
        assert normalise("cancel\u00adlation") == "cancellation"

    def test_bullets_start_their_own_line(self):
        raw = "targets: ● P1: 15 minutes ● P2: 1 hour"
        assert normalise(raw).splitlines() == ["targets:", "- P1: 15 minutes", "- P2: 1 hour"]

    def test_is_idempotent(self):
        # Ingest runs this once, but a second pass must not shift offsets or
        # the committed clause text would depend on how often it ran.
        once = normalise(extract_text(SOURCE_FILES[0].path))
        assert normalise(once) == once


class TestHeadingBreaks:
    def test_a_numbered_heading_starts_a_block(self):
        assert "\n\n2. Failed-pickup" in normalise("...no fee.\n2. Failed-pickup service credits")

    def test_a_heading_that_follows_a_sentence_mid_line_is_split(self):
        # The Northstar agreement runs "...applies. 3. Service credits" on one
        # visual line; without this the clause never gets its own block.
        out = normalise("the standard process applies. 3. Service credits")
        assert out.splitlines()[-1].startswith("3. Service credits")

    def test_a_decimal_amount_is_not_mistaken_for_a_heading(self):
        # "capped at INR 5,000. Unless this agreement..." must stay one block.
        out = normalise("Monthly credits are capped at INR 5,000. Unless this agreement states")
        assert "\n" not in out

    def test_a_known_issue_id_starts_a_block(self):
        out = normalise("creation is unaffected. KI-211 - SwiftShip pickup webhook delay")
        assert out.splitlines()[-1].startswith("KI-211")


class TestExtraction:
    def test_every_supplied_pdf_extracts_non_trivial_text(self):
        for source in SOURCE_FILES:
            text = normalise(extract_text(source.path))
            assert len(text) > 200, source.path.name

    def test_all_six_source_files_are_registered(self):
        assert len(SOURCE_FILES) == 6

    def test_every_registered_source_exists_on_disk(self):
        for source in SOURCE_FILES:
            assert source.path.is_file(), source.path

    def test_a_missing_file_raises_rather_than_returning_empty(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_text(tmp_path / "absent.pdf")

    @pytest.mark.parametrize(
        "filename,marker",
        [
            (
                "01_Support_Policy_v3_CURRENT.pdf",
                "Enterprise 30 minutes, 24x7 2 hours 1 business day",
            ),
            ("02_Support_Policy_v2_DEPRECATED.pdf", "Enterprise 1 hour 4 hours 2 business days"),
            ("03_Cancellation_and_Service_Credit_SOP_v4.pdf", "After 30 minutes"),
            ("04_Product_Operations_Guide_and_Known_Issues.pdf", "up to 5,000 rows per CSV"),
            ("05_Northstar_Logistics_Enterprise_Agreement.pdf", "no cancellation fee"),
            ("06_LumenWorks_Service_Agreement.pdf", "fixed INR 300 service credit"),
        ],
    )
    def test_the_load_bearing_sentence_of_each_document_survives(self, filename, marker):
        # One assertion per document, chosen as the phrase every downstream
        # answer depends on. If normalisation ever mangles one of these, the
        # failure points straight at the clause it broke.
        source = next(s for s in SOURCE_FILES if s.path.name == filename)
        assert marker in normalise(extract_text(source.path))
