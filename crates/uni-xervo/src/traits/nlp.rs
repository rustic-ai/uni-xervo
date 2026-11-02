//! Structured natural-language analysis types and traits.
//!
//! The trait is designed around multi-head cascade models (e.g.
//! `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`) that produce
//! token-level POS / NER / DEP tags, sentence segmentation, optional SRL
//! frames, and optional dialog-act classification in a single forward pass.

use crate::error::Result;
use crate::traits::ModelInfo;
use async_trait::async_trait;
use bitflags::bitflags;

bitflags! {
    /// Selects which NLP heads a caller wants populated.
    ///
    /// Implementations that support a head only do its post-processing
    /// (label decoding, dependency tree reconstruction, SRL frame
    /// assembly) when its flag is set — this saves measurable wall time
    /// on hot ingest paths where, for example, only NER is needed.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NlpTasks: u32 {
        /// Part-of-speech tagging.
        const POS = 1 << 0;
        /// Named entity recognition.
        const NER = 1 << 1;
        /// Dependency parsing.
        const DEP = 1 << 2;
        /// Semantic role labeling.
        const SRL = 1 << 3;
        /// Sentence-level classification (e.g. dialog acts).
        const CLS = 1 << 4;
        /// Universal-Dependencies morphological feature bundles, per word.
        ///
        /// kniv v7 only.
        const MORPH = 1 << 5;
        /// Lemmatisation, per word. kniv v7 only.
        const LEMMA = 1 << 6;
        /// Per-sentence sentiment (positive / negative / neutral). kniv v7 only.
        const SENTIMENT = 1 << 7;
        /// Coreference: mention clusters with span boundaries. kniv v7 only.
        const COREF = 1 << 8;
        /// Typed entities — the request schema's entity types, or the model's
        /// coarse types — grouped across their mentions. kniv v7 only.
        const ENTITIES = 1 << 9;
        /// Entity-to-entity facts against the request schema's relations, each
        /// with polarity, modality, holder and time. Implies
        /// [`ENTITIES`](Self::ENTITIES). kniv v7 only.
        const FACTS = 1 << 10;
        /// Convenience: the five heads of the v1 cascade (POS / NER / DEP /
        /// SRL / CLS).
        const CASCADE = Self::POS.bits() | Self::NER.bits() | Self::DEP.bits()
                      | Self::SRL.bits() | Self::CLS.bits();
        /// Convenience: every per-word and per-sentence linguistic head, with
        /// no entity or fact extraction.
        const LINGUISTIC = Self::CASCADE.bits() | Self::MORPH.bits()
                         | Self::LEMMA.bits() | Self::SENTIMENT.bits();
        /// Convenience: all currently-defined heads.
        const ALL = Self::LINGUISTIC.bits() | Self::COREF.bits()
                  | Self::ENTITIES.bits() | Self::FACTS.bits();
    }
}

impl NlpTasks {
    /// Normalise implied flags: [`FACTS`](Self::FACTS) implies
    /// [`ENTITIES`](Self::ENTITIES).
    #[must_use]
    pub fn expanded(self) -> Self {
        if self.contains(Self::FACTS) {
            self | Self::ENTITIES
        } else {
            self
        }
    }
}

/// One NLP analysis request: an input plus the heads to populate.
///
/// `#[non_exhaustive]`: build one with [`NlpRequest::new`] (or
/// [`NlpRequest::from_turns`]) and chain the `with_*` setters, so later
/// releases can add inputs without breaking callers.
///
/// ```
/// use uni_xervo::traits::{NlpRequest, NlpTasks};
/// let req = NlpRequest::new("Ada Lovelace was born in London.", NlpTasks::NER);
/// assert_eq!(req.tasks, NlpTasks::NER);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NlpRequest<'a> {
    /// Input prose. Borrowed so batch callers don't have to clone.
    ///
    /// kniv v7 splits it into paragraphs at blank lines (`"\n\n"`). Ignored
    /// when [`turns`](Self::turns) is set.
    pub text: &'a str,
    /// Which heads to populate. Unset flags result in empty fields on the
    /// corresponding [`NlpResult`] entries.
    pub tasks: NlpTasks,
    /// A conversation to analyse instead of [`text`](Self::text): one
    /// `(speaker, text)` [`Turn`] per message, oldest first.
    ///
    /// Each turn is its own unit — sentences never cross a turn — and its
    /// speaker becomes a `"speaker"` entity, so relations can be anchored on
    /// who said what. Models that do not support conversations ignore it.
    pub turns: Option<&'a [Turn<'a>]>,
    /// The relations and entity types to extract, for
    /// [`NlpTasks::ENTITIES`] / [`NlpTasks::FACTS`].
    ///
    /// Zero-shot: the schema is supplied per request and scored against the
    /// whole window. `None` leaves facts empty and falls back to the model's
    /// own coarse entity types.
    pub schema: Option<&'a ExtractionSchema>,
    /// Per-request overrides of the model's shipped decoding thresholds.
    ///
    /// `None` uses the model's `calibration.json` as published, which is
    /// what the reported quality figures were measured with.
    pub calibration: Option<&'a NlpCalibrationOverrides>,
}

impl<'a> NlpRequest<'a> {
    /// A request over prose.
    #[must_use]
    pub fn new(text: &'a str, tasks: NlpTasks) -> Self {
        Self {
            text,
            tasks,
            turns: None,
            schema: None,
            calibration: None,
        }
    }

    /// A request over conversation turns.
    #[must_use]
    pub fn from_turns(turns: &'a [Turn<'a>], tasks: NlpTasks) -> Self {
        Self {
            text: "",
            tasks,
            turns: Some(turns),
            schema: None,
            calibration: None,
        }
    }

    /// Attach conversation turns, which take precedence over
    /// [`text`](Self::text).
    #[must_use]
    pub fn with_turns(mut self, turns: &'a [Turn<'a>]) -> Self {
        self.turns = Some(turns);
        self
    }

    /// Attach the extraction schema.
    #[must_use]
    pub fn with_schema(mut self, schema: &'a ExtractionSchema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Attach calibration overrides.
    #[must_use]
    pub fn with_calibration(mut self, calibration: &'a NlpCalibrationOverrides) -> Self {
        self.calibration = Some(calibration);
        self
    }
}

/// One turn of a conversation passed as [`NlpRequest::turns`].
#[derive(Debug, Clone, Copy)]
pub struct Turn<'a> {
    /// Who is speaking, e.g. `"user"` or `"assistant"`. Turns by the same
    /// speaker share one `"speaker"` entity.
    pub speaker: &'a str,
    /// What they said.
    pub text: &'a str,
}

impl<'a> Turn<'a> {
    /// A turn.
    #[must_use]
    pub fn new(speaker: &'a str, text: &'a str) -> Self {
        Self { speaker, text }
    }
}

/// The relations and entity types to extract from one request.
///
/// Schema label embeddings are computed once and cached per exact label list,
/// so a stable schema costs nothing after the first request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExtractionSchema {
    /// Relations to score for every ordered entity pair. Empty means no
    /// facts are extracted.
    pub relations: Vec<LabelSpec>,
    /// Entity types to extract. `None` falls back to the model's coarse
    /// types (kniv v7: `LOC`, `MISC`, `NUM`, `ORG`, `PER`, `TIME`).
    pub entity_types: Option<Vec<LabelSpec>>,
    /// How many relations to keep per entity pair. `None` uses the model's
    /// calibrated `rel_top` (kniv v7: 4).
    pub top_per_pair: Option<usize>,
}

/// One schema label: a name, and a description that tells the model what it
/// means and which side of a relation is the head.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LabelSpec {
    /// The label as a reader would name it, e.g. `"lives in"`. This is the
    /// string reported back on [`NlpFact::relation`] / [`NlpEntity::ty`].
    pub name: String,
    /// A short gloss, e.g. `"the place where the head resides"`.
    pub description: Option<String>,
}

impl LabelSpec {
    /// A label with no description.
    #[must_use]
    pub fn name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
        }
    }

    /// A label with a description.
    #[must_use]
    pub fn described(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
        }
    }
}

/// Structured output of an NLP analysis call.
///
/// Field population depends on the [`NlpTasks`] flags in the request and on
/// the model's [`NlpModel::supported_tasks`]. Unrequested or unsupported
/// fields are empty (`Vec::new()`).
///
/// This struct is `#[non_exhaustive]`: construct it via [`NlpResult::default`]
/// and assign fields, so future heads can be added without breaking callers.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct NlpResult {
    /// Token sequence with optional POS / NER / DEP annotations.
    pub tokens: Vec<NlpToken>,
    /// Sentence boundaries (always populated when any head is requested).
    pub sentences: Vec<NlpSentence>,
    /// SRL frames. Empty unless [`NlpTasks::SRL`] was requested and supported.
    pub frames: Vec<SrlFrame>,
    /// Sentence-level classifications. Empty unless [`NlpTasks::CLS`] was
    /// requested and supported.
    pub speech_acts: Vec<SpeechAct>,
    /// Merged named-entity spans, a BIO-collapsed view of the per-token
    /// [`NlpToken::ner`] tags. Empty unless [`NlpTasks::NER`] was requested
    /// and supported. See [`NerEntity`].
    pub entities: Vec<NerEntity>,
    /// The full per-window analysis, for models that analyse whole windows
    /// of words rather than one token sequence (kniv v7).
    ///
    /// One entry per window, in input order; the fields above are a
    /// flattened, v1-shaped projection of the same content, so callers that
    /// only read them keep working. Empty for models without a window
    /// pipeline.
    pub windows: Vec<NlpWindow>,
}

/// One analysis window: a run of whole sentences of up to the model's word
/// budget, analysed in a single pass.
///
/// Windows never cross a paragraph or a turn, and a sentence is never split
/// across two windows. Every index in this struct — heads, spans, mention
/// positions — is a **word** index into [`tokens`](Self::tokens), local to
/// the window.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct NlpWindow {
    /// The window's words, in order.
    pub tokens: Vec<String>,
    /// How many leading words fit in the encoder's subword budget.
    ///
    /// Words from this index on are reported in [`tokens`](Self::tokens) but
    /// carry no analysis. Equal to `tokens.len()` for a window that fits.
    pub analysed_words: usize,
    /// The window's sentences.
    pub sentences: Vec<WindowSentence>,
    /// Per-word analysis, one entry per analysed word (so
    /// `words.len() == analysed_words`).
    pub words: Vec<WordAnalysis>,
    /// Named entities as `(type, start, end)` half-open word spans, using
    /// the model's own NER vocabulary (kniv v7: OntoNotes).
    pub ner: Vec<WordSpan>,
    /// Semantic-role frames, one per predicate — every predicate, not only
    /// verbs.
    pub srl: Vec<WindowFrame>,
    /// Coreference clusters: each cluster is a list of **inclusive**
    /// `(start, end)` word spans.
    pub coref: Vec<Vec<(usize, usize)>>,
    /// Typed entities, grouped across their mentions. In a conversation the
    /// speakers come first, in order of first appearance.
    pub entities: Vec<NlpEntity>,
    /// Facts between entities, scored against the request schema's
    /// relations.
    pub facts: Vec<NlpFact>,
}

/// A labelled half-open `[start, end)` span of words.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WordSpan {
    /// The label, e.g. an entity type or a semantic role.
    pub label: String,
    /// First word index, inclusive.
    pub start: usize,
    /// Last word index, exclusive.
    pub end: usize,
}

/// One sentence of an [`NlpWindow`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct WindowSentence {
    /// First word index, inclusive.
    pub start: usize,
    /// Last word index, exclusive.
    pub end: usize,
    /// Which input unit (paragraph, or turn) this sentence came from.
    pub unit: usize,
    /// The speaker of that unit, for a conversation.
    pub speaker: Option<String>,
    /// Dialogue acts above threshold. Multi-label: a sentence can be both
    /// an answer and a question.
    pub cls: Vec<String>,
    /// Per-class dialogue-act probabilities, parallel to
    /// [`NlpLabelMaps::cls`].
    pub cls_scores: Vec<f32>,
    /// Sentiment label, e.g. `"positive"`.
    pub sentiment: Option<String>,
    /// Whether the sentence ends within [`NlpWindow::analysed_words`]. A
    /// truncated sentence's analysis is partial.
    pub whole: bool,
}

/// Per-word linguistic analysis.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct WordAnalysis {
    /// UD UPOS tag.
    pub pos: Option<String>,
    /// UD morphological feature bundle, e.g.
    /// `"Mood=Ind|Number=Sing|Tense=Past"`. `"_"` when the word has none.
    pub morph: Option<String>,
    /// The lemma.
    pub lemma: Option<String>,
    /// Dependency head: `0` is the sentence root, otherwise the 1-based word
    /// index of the head within the window. `None` when DEP was not
    /// requested.
    pub head: Option<usize>,
    /// UD dependency relation to [`head`](Self::head).
    pub deprel: Option<String>,
}

/// One semantic-role frame of an [`NlpWindow`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct WindowFrame {
    /// Word index of the predicate.
    pub predicate: usize,
    /// Argument spans, excluding the predicate itself.
    pub args: Vec<WordSpan>,
}

/// One typed entity of an [`NlpWindow`], grouped across its mentions.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct NlpEntity {
    /// The entity type: a name from [`ExtractionSchema::entity_types`] when
    /// one was given, else the model's coarse type, or `"speaker"` for a
    /// conversation participant.
    pub ty: String,
    /// Surface text of the entity's first mention; for a speaker, the
    /// speaker name.
    pub text: String,
    /// Whether this entity is a conversation participant rather than a
    /// mention in the text.
    pub speaker: bool,
    /// Surface text of each mention, parallel to [`spans`](Self::spans).
    pub mentions: Vec<String>,
    /// Half-open `[start, end)` word spans of the mentions. For a speaker,
    /// the turn-start positions where they begin a turn.
    pub spans: Vec<(usize, usize)>,
}

/// One extracted fact: a relation between two entities, with its attributes.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct NlpFact {
    /// Index into [`NlpWindow::entities`] of the head (subject).
    pub head: usize,
    /// Index into [`NlpWindow::entities`] of the tail (object).
    pub tail: usize,
    /// The relation name, as given in [`ExtractionSchema::relations`].
    pub relation: String,
    /// Calibrated margin over the model's keep threshold. Always `> 0` for a
    /// reported fact; larger is surer.
    ///
    /// Not a probability. Map it to one with `sigmoid` if a confidence is
    /// wanted, and tune a relation's threshold through
    /// [`NlpCalibrationOverrides::rel_offsets`] rather than by filtering on
    /// the margin.
    pub margin: f32,
    /// Whether the fact is asserted or denied.
    pub polarity: FactPolarity,
    /// How the fact is held: asserted, planned, merely questioned, and so
    /// on.
    pub modality: FactModality,
    /// Index into [`NlpWindow::entities`] of whoever holds this belief, when
    /// the model found one — the speaker for `"I live in Lisbon"`, the
    /// reporter for `"Leon says I work at Northstar"`.
    pub holder: Option<usize>,
    /// Index into [`NlpWindow::entities`] of a temporal entity anchoring the
    /// fact. Only ever set when the schema has temporal entity types.
    pub time: Option<usize>,
}

/// Whether a [`NlpFact`] is asserted or denied.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum FactPolarity {
    /// The fact is claimed to hold.
    #[default]
    Positive,
    /// The fact is claimed not to hold — `"I don't work there anymore"`.
    Negative,
}

impl FactPolarity {
    /// The reference runtime's lowercase name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Negative => "negative",
        }
    }
}

/// How a [`NlpFact`] is held by its holder.
///
/// Only [`Asserted`](Self::Asserted) and [`Reported`](Self::Reported) are
/// claims about the world; the rest are intentions, questions and
/// suppositions, and consolidating them into current state is a mistake.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum FactModality {
    /// Stated as fact.
    #[default]
    Asserted,
    /// Conditional or counterfactual — `"if I moved to Lisbon"`.
    Hypothetical,
    /// An intention with a plan — `"next month I'm joining Northstar"`.
    Planned,
    /// A wish — `"I'd love to live in Lisbon"`.
    Desired,
    /// Asked, not stated — `"do you live in Lisbon?"`.
    Questioned,
    /// Attributed to someone else — `"Leon says I work at Northstar"`.
    Reported,
}

impl FactModality {
    /// The reference runtime's lowercase name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Asserted => "asserted",
            Self::Hypothetical => "hypothetical",
            Self::Planned => "planned",
            Self::Desired => "desired",
            Self::Questioned => "questioned",
            Self::Reported => "reported",
        }
    }
}

/// One token in an [`NlpResult`].
///
/// Tokens are subword units; several may belong to one word (see
/// [`word_index`](NlpToken::word_index)). Index-based references elsewhere
/// (e.g. [`DepLink::head`], [`SrlRole::span`]) are always *token* indices into
/// [`NlpResult::tokens`], not word indices.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct NlpToken {
    /// Surface form of the token.
    pub text: String,
    /// UTF-8 byte offset of the token start in the original input text.
    pub start: usize,
    /// UTF-8 byte offset of the token end (exclusive).
    pub end: usize,
    /// POS tag (typically Universal Dependencies tagset). `None` if POS not requested.
    pub pos: Option<String>,
    /// Named-entity type (e.g. `"PERSON"`, `"ORG"`). `None` outside any entity span.
    pub ner: Option<String>,
    /// Dependency-parse head and relation label. `None` if DEP not requested.
    pub dep: Option<DepLink>,
    /// Index of the word this subword token belongs to.
    ///
    /// Dense and monotonically non-decreasing over [`NlpResult::tokens`]:
    /// consecutive tokens that form one word share a `word_index`, and each new
    /// word increments it by one starting at `0`. Lets consumers regroup
    /// subwords into words without parsing tokenizer metaspace markers.
    pub word_index: usize,
}

/// One sentence in an [`NlpResult`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct NlpSentence {
    /// Inclusive `[first, last]` token indices into [`NlpResult::tokens`].
    pub token_range: (usize, usize),
    /// UTF-8 byte offset of the sentence start in the original text.
    pub start: usize,
    /// UTF-8 byte offset of the sentence end (exclusive).
    pub end: usize,
}

/// A dependency-parse arc attached to an [`NlpToken`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DepLink {
    /// Syntactic head as a 0-based index into [`NlpResult::tokens`].
    ///
    /// `Some(i)` points at the head token; `None` means this token attaches to
    /// the sentence root (it has no head). The index is global across the whole
    /// [`NlpResult`], not chunk- or sentence-local.
    pub head: Option<usize>,
    /// Relation label (e.g. `"nsubj"`, `"obj"`, `"amod"`).
    pub relation: String,
}

/// One semantic-role-labeling frame: a predicate and its argument spans.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SrlFrame {
    /// 0-based index into [`NlpResult::tokens`] of the predicate token.
    ///
    /// The predicate token is never itself included in [`roles`](SrlFrame::roles).
    pub predicate_token: usize,
    /// Predicate sense identifier when known (e.g. `"buy.01"`).
    pub predicate_sense: Option<String>,
    /// Argument spans, excluding the predicate token.
    pub roles: Vec<SrlRole>,
}

/// One argument span of an [`SrlFrame`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SrlRole {
    /// Inclusive `[first, last]` 0-based token indices into [`NlpResult::tokens`].
    ///
    /// Both endpoints are inclusive; a single-token argument has `first == last`.
    pub span: (usize, usize),
    /// Role label (e.g. `"ARG0"`, `"ARG1"`, `"ARGM-TMP"`).
    pub label: String,
}

/// One sentence-level classification result.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SpeechAct {
    /// Index into [`NlpResult::sentences`].
    pub sentence_index: usize,
    /// Class label (e.g. `"STATEMENT"`, `"QUESTION"`, `"GREETING"`).
    ///
    /// Equals the label of the argmax of [`scores`](SpeechAct::scores).
    pub label: String,
    /// Provider-reported confidence in `[0.0, 1.0]`.
    ///
    /// Equals `scores[argmax]` — the probability of the winning [`label`](SpeechAct::label).
    pub confidence: f32,
    /// Full per-class softmax over the model's dialog-act vocabulary.
    ///
    /// Parallel to [`NlpLabelMaps::cls`]: `scores[i]` is the probability of the
    /// class named `cls[i]`, and the values sum to ~1.0. Empty when the model
    /// does not expose a distribution. Lets consumers do thresholded
    /// multi-label gating instead of relying on the top-1 [`label`](SpeechAct::label).
    pub scores: Vec<f32>,
}

/// One merged named-entity span, a BIO-collapsed view of per-token NER tags.
///
/// Produced by collapsing the per-token [`NlpToken::ner`] BIO tags: a `B-` (or
/// orphan `I-`) tag opens a span, matching `I-` tags extend it, and an `O` tag
/// or a label change closes it. Spans never include the `"O"` outside-tag.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct NerEntity {
    /// Surface text of the entity, spanning the original input bytes.
    pub text: String,
    /// Entity type without the BIO prefix (e.g. `"PERSON"`, `"GPE"`).
    pub label: String,
    /// Inclusive `[first, last]` 0-based token indices into [`NlpResult::tokens`].
    pub token_span: (usize, usize),
    /// Half-open `[start, end)` UTF-8 byte offsets into the original input text.
    pub char_span: (usize, usize),
}

/// The label vocabularies a model decodes against, one list per head.
///
/// Each vector is indexed by the model's internal class id, so a decoded class
/// id `k` for a head maps to `labels.<head>[k]`. In particular
/// [`cls`](NlpLabelMaps::cls) is parallel to [`SpeechAct::scores`]. Exposing
/// these lets consumers map labels back to ids without embedding a parallel
/// copy of the model's `label_maps.json`.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct NlpLabelMaps {
    /// Part-of-speech tag vocabulary.
    pub pos: Vec<String>,
    /// Named-entity BIO tag vocabulary.
    pub ner: Vec<String>,
    /// Dependency relation vocabulary.
    pub deprel: Vec<String>,
    /// Semantic-role BIO tag vocabulary.
    pub srl: Vec<String>,
    /// Dialog-act / sentence-classification vocabulary, parallel to
    /// [`SpeechAct::scores`].
    pub cls: Vec<String>,
    /// Morphological feature-bundle vocabulary. Empty for models without a
    /// morphology head.
    pub morph: Vec<String>,
    /// Lemmatiser vocabulary. kniv v7 predicts an edit script
    /// (`"case|strip|append"`) rather than the lemma itself, so these are
    /// scripts, not words.
    pub lemma: Vec<String>,
    /// Sentiment vocabulary. Empty for models without a sentiment head.
    pub sentiment: Vec<String>,
    /// Coarse entity-type vocabulary used by the default entity head, i.e.
    /// what [`NlpEntity::ty`] carries when the request has no
    /// [`ExtractionSchema::entity_types`].
    pub ent_type: Vec<String>,
}

/// The decoding thresholds a model ships in `calibration.json`.
///
/// These are not hyperparameters of the graphs; they were tuned on dev data
/// and are what the published quality figures were measured with. Override
/// them per request via [`NlpRequest::calibration`] only against a labelled
/// sample.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct NlpCalibration {
    /// Per-relation offsets, by relation name. A fact is kept when its raw
    /// margin exceeds `-offset`.
    pub rel_offsets: std::collections::HashMap<String, f32>,
    /// The offset applied to relations the model was not calibrated for,
    /// as `-unseen_offset`.
    pub unseen_offset: f32,
    /// A global offset added to every relation threshold.
    pub offset: f32,
    /// How many facts to keep per entity pair.
    pub rel_top: usize,
    /// Subtracted from entity-type span margins.
    pub span_offset: f32,
    /// Added to the "no antecedent" coreference score. Higher links less.
    pub coref_link_bias: f32,
    /// Added to the `O` column of the default entity head.
    pub ent_o_bias: f32,
    /// How coreference groups entity mentions: `"none"`, `"compatible"` or
    /// `"all"`. Both kniv v7 models ship `"none"` — chains merge mentions
    /// too eagerly and cost end-to-end relation F1.
    pub entity_coref: String,
}

/// Per-request overrides of [`NlpCalibration`]. Unset fields keep the
/// model's shipped value.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct NlpCalibrationOverrides {
    /// Replaces individual entries of [`NlpCalibration::rel_offsets`],
    /// leaving the rest as shipped.
    pub rel_offsets: std::collections::HashMap<String, f32>,
    /// Overrides [`NlpCalibration::unseen_offset`].
    pub unseen_offset: Option<f32>,
    /// Overrides [`NlpCalibration::span_offset`].
    pub span_offset: Option<f32>,
    /// Overrides [`NlpCalibration::coref_link_bias`].
    pub coref_link_bias: Option<f32>,
    /// Overrides [`NlpCalibration::ent_o_bias`].
    pub ent_o_bias: Option<f32>,
    /// Overrides [`NlpCalibration::entity_coref`].
    pub entity_coref: Option<String>,
}

impl NlpCalibration {
    /// This calibration with `overrides` applied.
    #[must_use]
    pub fn with_overrides(&self, overrides: &NlpCalibrationOverrides) -> Self {
        let mut out = self.clone();
        for (name, value) in &overrides.rel_offsets {
            out.rel_offsets.insert(name.clone(), *value);
        }
        if let Some(v) = overrides.unseen_offset {
            out.unseen_offset = v;
        }
        if let Some(v) = overrides.span_offset {
            out.span_offset = v;
        }
        if let Some(v) = overrides.coref_link_bias {
            out.coref_link_bias = v;
        }
        if let Some(v) = overrides.ent_o_bias {
            out.ent_o_bias = v;
        }
        if let Some(v) = &overrides.entity_coref {
            out.entity_coref = v.clone();
        }
        out
    }
}

/// A multi-head structured-NLP model.
///
/// # Canonical default model
///
/// The reference implementation is
/// [`dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-xsmall):
/// a DeBERTa-v3-xsmall multi-head cascade producing POS / NER / DEP / SRL / CLS
/// in a single forward pass. It will ship as the canonical `nlp/default`
/// catalog alias once the `local/onnx` provider gains an `NlpModel`
/// implementation in a follow-up release.
///
/// Tagsets used by the reference model:
/// - POS: Universal Dependencies English EWT (17 classes)
/// - NER: OntoNotes (18 entity types)
/// - DEP: Universal Dependencies English EWT
/// - SRL: PropBank (42 classes)
/// - CLS: 8 dialog acts (internal dataset)
///
/// Language: English only. Maximum input: 128 tokens — provider-side
/// chunking is the implementation's responsibility; the trait contract
/// keeps token offsets stable in whole-document UTF-8 byte coordinates
/// regardless of internal chunking.
#[async_trait]
pub trait NlpModel: ModelInfo {
    /// Analyze a batch of texts, returning one [`NlpResult`] per request.
    ///
    /// # Errors
    /// Returns an error if a requested task is not in
    /// [`supported_tasks`](NlpModel::supported_tasks), or if the provider
    /// fails internally.
    async fn analyze(&self, requests: Vec<NlpRequest<'_>>) -> Result<Vec<NlpResult>>;

    /// Heads this model can populate. Callers may request a subset via
    /// [`NlpRequest::tasks`].
    fn supported_tasks(&self) -> NlpTasks;

    /// The label vocabularies this model decodes against, if it exposes them.
    ///
    /// Returns `Some` for models with a fixed, introspectable label set (e.g.
    /// the ONNX cascade), letting callers map labels to class ids and read
    /// [`SpeechAct::scores`] against [`NlpLabelMaps::cls`]. The default is
    /// `None` for models that do not expose vocabularies (e.g. remote or mock).
    fn label_maps(&self) -> Option<&NlpLabelMaps> {
        None
    }

    /// The decoding thresholds this model ships, if it has any.
    ///
    /// `Some` for kniv v7, whose `calibration.json` is what
    /// [`NlpRequest::calibration`] overrides.
    fn calibration(&self) -> Option<&NlpCalibration> {
        None
    }

    /// Optional warmup hook. The default is a no-op.
    ///
    /// # Errors
    /// Returns an error if the underlying model fails to initialize.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn nlp_model_is_dyn_safe() {
        fn _accept(_: Arc<dyn NlpModel>) {}
    }

    #[test]
    fn nlp_tasks_all_includes_every_head() {
        let all = NlpTasks::ALL;
        for flag in [
            NlpTasks::POS,
            NlpTasks::NER,
            NlpTasks::DEP,
            NlpTasks::SRL,
            NlpTasks::CLS,
            NlpTasks::MORPH,
            NlpTasks::LEMMA,
            NlpTasks::SENTIMENT,
            NlpTasks::COREF,
            NlpTasks::ENTITIES,
            NlpTasks::FACTS,
        ] {
            assert!(all.contains(flag), "{flag:?} missing from NlpTasks::ALL");
        }
    }

    #[test]
    fn facts_imply_entities() {
        assert!(NlpTasks::FACTS.expanded().contains(NlpTasks::ENTITIES));
        assert!(!NlpTasks::POS.expanded().contains(NlpTasks::ENTITIES));
    }

    #[test]
    fn calibration_overrides_only_touch_set_fields() {
        let base = crate::traits::NlpCalibration {
            unseen_offset: -1.5,
            span_offset: -2.0,
            rel_top: 4,
            entity_coref: "none".to_string(),
            ..Default::default()
        };
        let overrides = crate::traits::NlpCalibrationOverrides {
            span_offset: Some(-1.0),
            ..Default::default()
        };
        let merged = base.with_overrides(&overrides);
        assert_eq!(merged.span_offset, -1.0);
        assert_eq!(merged.unseen_offset, -1.5);
        assert_eq!(merged.rel_top, 4);
        assert_eq!(merged.entity_coref, "none");
    }
}
