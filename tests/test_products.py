"""Transcript formats, keywords, notes, action items, search and their metrics."""

from __future__ import annotations

import pytest

from echo.asr.base import Segment, TranscriptionResult, Word
from echo.corpus.build import UtteranceLabel
from echo.corpus.scripts import lecture_by_id, meeting_by_id
from echo.metrics.products import (
    Confusion,
    Counts,
    key_sentence_recall,
    score_actions,
    score_query,
    score_search,
    token_overlap,
)
from echo.products.actions import detect_actions, extract_due, extract_owner, score_sentence
from echo.products.keywords import extract_keywords, is_content, keyword_hits
from echo.products.notes import (
    Bullet,
    build_notes,
    join_words,
    notes_to_markdown,
    section_breaks,
    split_sentences,
    textrank,
    tfidf_matrix,
)
from echo.products.search import SearchIndex
from echo.products.stopwords import STOPWORDS, is_verb_like
from echo.products.transcript import assign_speakers, format_timestamp, to_dict, to_srt, to_text, to_vtt
from echo.vad.energy import SpeechRegion
from tests.conftest import make_transcript


def test_timestamp_formats():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.5) == "01:01:01,500"
    assert format_timestamp(1.2345, ".") == "00:00:01.234"
    assert format_timestamp(-1) == "00:00:00,000"


def test_assign_speakers_and_exports():
    words = (Word("Hello", 0.0, 0.4), Word("there.", 0.4, 0.8))
    result = TranscriptionResult(
        (Segment(0, 0.0, 0.8, "Hello there.", words), Segment(1, 2.0, 2.9, "Hi.", (Word("Hi.", 2.0, 2.9),))),
        "en",
        "test",
        "m",
        0.1,
        3.0,
    )
    regions = [SpeechRegion(0.0, 0.9), SpeechRegion(1.9, 3.0)]
    transcript = assign_speakers(result, regions, (0, 1))
    assert transcript.speakers == ("Speaker 1", "Speaker 2")
    srt = to_srt(transcript)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:00,800\nSpeaker 1: Hello there.")
    vtt = to_vtt(transcript)
    assert vtt.startswith("WEBVTT") and "<v Speaker 2>Hi." in vtt
    text = to_text(transcript)
    assert "Speaker 1:" in text and "Speaker 2:" in text
    payload = to_dict(transcript)
    assert payload["segments"][0]["words"][0]["text"] == "Hello"
    assert transcript.text == "Hello there. Hi."
    assert len(transcript.words) == 3


def test_assign_speakers_without_regions_defaults_to_first_speaker():
    result = TranscriptionResult((Segment(0, 0.0, 1.0, "x", ()),), "en", "t", "m", 0.0, 1.0)
    transcript = assign_speakers(result, [], ())
    assert transcript.segments[0].speaker == "Speaker 1"


def test_keyword_extraction_on_lecture(lecture_transcript):
    keywords = extract_keywords(lecture_transcript, top_k=10)
    assert len(keywords) == 10
    assert keywords == sorted(keywords, key=lambda k: -k.score)
    phrases = {k.phrase for k in keywords}
    assert "carbon dioxide" in phrases
    assert all(k.timestamps and k.count == len(k.timestamps) for k in keywords)
    assert keyword_hits(keywords, lecture_by_id("photosynthesis").key_terms) >= 3
    assert extract_keywords(make_transcript(["the of and"]), 5) == []


def test_content_filters():
    assert not is_content("the") and not is_content("42") and not is_content("quickly") and not is_content("describes")
    assert is_content("chlorophyll")
    assert is_verb_like("running") and is_verb_like("stopped") and is_verb_like("uses")
    assert not is_verb_like("photosynthesis")
    assert "okay" in STOPWORDS


def test_sentence_split_across_segments_and_spacing():
    words_a = (Word("Light", 0.0, 0.2), Word("-dependent", 0.2, 0.5, prefix_space=False), Word("reactions", 0.5, 0.8))
    words_b = (Word("release", 1.0, 1.3), Word("oxygen.", 1.3, 1.6), Word("Done.", 1.7, 2.0))
    result = TranscriptionResult(
        (
            Segment(0, 0.0, 0.8, "Light-dependent reactions", words_a),
            Segment(1, 1.0, 2.0, "release oxygen. Done.", words_b),
        ),
        "en",
        "t",
        "m",
        0.0,
        2.0,
    )
    transcript = assign_speakers(result, [SpeechRegion(0, 2)], (0,))
    sentences = split_sentences(transcript)
    assert [s.text for s in sentences] == ["Light-dependent reactions release oxygen.", "Done."]
    assert sentences[0].start == 0.0 and sentences[0].end == pytest.approx(1.6)
    assert join_words([]) == ""
    no_words = assign_speakers(
        TranscriptionResult((Segment(0, 0, 1, "Plain segment", ()),), "en", "t", "m", 0, 1), [], ()
    )
    assert split_sentences(no_words)[0].text == "Plain segment"


def test_textrank_and_tfidf_are_deterministic_and_sane(lecture_transcript):
    sentences = split_sentences(lecture_transcript)
    matrix = tfidf_matrix(sentences)
    assert matrix.shape[0] == 10
    scores = textrank(matrix @ matrix.T)
    assert scores.shape == (10,) and scores.sum() == pytest.approx(1.0, abs=1e-6)
    assert (textrank(matrix @ matrix.T) == scores).all()
    assert textrank(matrix[:0] @ matrix[:0].T).shape == (0,)
    assert isinstance(section_breaks(matrix), list)


def test_build_notes_keeps_key_sentences(lecture_transcript):
    lecture = lecture_by_id("photosynthesis")
    notes = build_notes(lecture_transcript, extract_keywords(lecture_transcript))
    assert 3 <= len(notes.summary) <= 8
    assert notes.n_sentences == 10
    assert 0 < notes.compression <= 1
    assert [b.start for b in notes.summary] == sorted(b.start for b in notes.summary)
    gold = [lecture.sentences[i] for i in lecture.key_sentences]
    assert key_sentence_recall(notes.summary, gold).hits >= 2
    markdown = notes_to_markdown(notes, "Photosynthesis")
    assert markdown.startswith("# Photosynthesis") and "## Summary" in markdown
    assert all(section.bullets for section in notes.sections)
    assert build_notes(make_transcript([]), []).n_sentences == 0


def test_action_cues_owner_and_due():
    score, cues = score_sentence("Tom, please rewrite the rubric and share it by Thursday.")
    assert score >= 2.0 and "please" in cues and "deadline" in cues
    assert score_sentence("Thanks everyone, that covers the agenda.")[0] < 2.0
    assert extract_owner("Tom, please rewrite the rubric.", "Speaker 1") == "Tom"
    assert extract_owner("Yes, I will send them tomorrow morning.", "Speaker 3") == "Speaker 3"
    assert extract_owner("We should also update the course page.", "Speaker 2") == "team"
    assert extract_owner("Lena will draft the runbook.", "Speaker 1") == "Lena"
    assert extract_owner("Can you make sure the link is in the next build?", "Speaker 1") == "unassigned"
    assert extract_due("confirm by Monday") == "by Monday"
    assert extract_due("by the end of the week") == "by the end of the week"
    assert extract_due("I will send them tomorrow morning.") == "tomorrow morning"
    assert extract_due("no deadline here") is None


def test_detect_actions_on_meeting(meeting_transcript):
    meeting = meeting_by_id("course_staff")
    actions = detect_actions(meeting_transcript)
    utterances = [
        UtteranceLabel(i, t.speaker, "v", t.text, seg.start, seg.end, t.action, t.owner, t.due)
        for i, (t, seg) in enumerate(zip(meeting.turns, meeting_transcript.segments, strict=True))
    ]
    confusion, owners = score_actions(actions, utterances)
    assert confusion.recall >= 0.7
    assert confusion.precision >= 0.8
    assert owners.total > 0 and owners.ratio >= 0.6
    assert all(a.start <= a.end for a in actions)


def test_search_index_ranks_the_answer(meeting_transcript, lecture_transcript):
    index = SearchIndex.from_transcript(meeting_transcript)
    hits = index.search("plagiarism checker", k=3)
    assert hits and "plagiarism" in hits[0].text.lower()
    assert hits[0].score >= (hits[1].score if len(hits) > 1 else 0)
    assert index.search("", k=3) == [] and index.search("zzzz qqqq", k=3) == []
    assert index.idf("plagiarism") > index.idf("question")
    lecture_index = SearchIndex.from_transcript(lecture_transcript)
    lecture = lecture_by_id("photosynthesis")
    for query in lecture.queries:
        gold = lecture.sentences[query.answer_index]
        assert any(token_overlap(h.text, gold) > 0.8 for h in lecture_index.search(query.text, 3))
    assert SearchIndex([]).search("anything") == []


def test_product_metric_helpers():
    assert token_overlap("the cat sat", "the cat sat") == 1.0
    assert token_overlap("", "x") == 0.0
    counts = Counts(1, 4) + Counts(1, 4)
    assert counts.ratio == 0.25 and Counts(0, 0).ratio == 0.0
    confusion = Confusion(2, 1, 1)
    assert confusion.precision == pytest.approx(2 / 3) and confusion.recall == pytest.approx(2 / 3)
    assert confusion.f1 == pytest.approx(2 / 3)
    assert Confusion(0, 0, 0).f1 == 0.0
    bullets = [Bullet("The water cycle describes how water moves.", 0.0, "s", 1.0)]
    assert key_sentence_recall(bullets, ["The water cycle describes how water moves between the oceans."]).hits == 1
    gold = UtteranceLabel(0, "A", "v", "t", 10.0, 12.0)
    from echo.products.search import Hit

    hits = [Hit(0, 1.0, 0.0, 1.0, "x", "s"), Hit(1, 0.5, 10.5, 11.0, "y", "s")]
    assert score_query(hits, gold) == 2
    assert score_query([], gold) == 0
    search = score_search([1, 2, 0, 1])
    assert search.hit_at_1.ratio == 0.5 and search.hit_at_3.ratio == 0.75
    assert search.mrr == pytest.approx((1 + 0.5 + 0 + 1) / 4)
