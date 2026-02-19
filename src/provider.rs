//! Provider implementations for local and remote model backends.
//!
//! Each sub-module is gated behind a Cargo feature flag (e.g. `provider-candle`,
//! `provider-openai`). Only providers whose features are enabled will be compiled.
//!
//! ## Local providers
//!
//! | Module | Feature | Engine |
//! |--------|---------|--------|
//! | `candle` | `provider-candle` | [Candle](https://github.com/huggingface/candle) |
//! | `fastembed` | `provider-fastembed` | [FastEmbed](https://github.com/Anush008/fastembed-rs) (ONNX Runtime) |
//! | `mistralrs` | `provider-mistralrs` | [mistral.rs](https://github.com/EricLBuehler/mistral.rs) |
//!
//! ## Remote providers
//!
//! | Module | Feature | API |
//! |--------|---------|-----|
//! | `openai` | `provider-openai` | OpenAI |
//! | `gemini` | `provider-gemini` | Google Gemini |
//! | `vertexai` | `provider-vertexai` | Google Vertex AI |
//! | `mistral` | `provider-mistral` | Mistral AI |
//! | `anthropic` | `provider-anthropic` | Anthropic |
//! | `voyageai` | `provider-voyageai` | Voyage AI |
//! | `cohere` | `provider-cohere` | Cohere |
//! | `azure_openai` | `provider-azure-openai` | Azure OpenAI |

#[cfg(feature = "provider-candle")]
pub mod candle;

#[cfg(any(
    feature = "provider-openai",
    feature = "provider-gemini",
    feature = "provider-vertexai",
    feature = "provider-mistral",
    feature = "provider-anthropic",
    feature = "provider-voyageai",
    feature = "provider-cohere",
    feature = "provider-azure-openai",
))]
pub(crate) mod remote_common;

#[cfg(feature = "provider-openai")]
pub mod openai;

#[cfg(feature = "provider-fastembed")]
pub mod fastembed;

#[cfg(feature = "provider-gemini")]
pub mod gemini;

#[cfg(feature = "provider-vertexai")]
pub mod vertexai;

#[cfg(feature = "provider-mistralrs")]
pub mod mistralrs;

#[cfg(feature = "provider-mistral")]
pub mod mistral;

#[cfg(feature = "provider-anthropic")]
pub mod anthropic;

#[cfg(feature = "provider-voyageai")]
pub mod voyageai;

#[cfg(feature = "provider-cohere")]
pub mod cohere;

#[cfg(feature = "provider-azure-openai")]
pub mod azure_openai;

// Re-exports (same order as module declarations above).
#[cfg(feature = "provider-candle")]
pub use candle::LocalCandleProvider;

#[cfg(feature = "provider-openai")]
pub use openai::RemoteOpenAIProvider;

#[cfg(feature = "provider-fastembed")]
pub use fastembed::LocalFastEmbedProvider;

#[cfg(feature = "provider-gemini")]
pub use gemini::RemoteGeminiProvider;

#[cfg(feature = "provider-vertexai")]
pub use vertexai::RemoteVertexAIProvider;

#[cfg(feature = "provider-mistralrs")]
pub use self::mistralrs::LocalMistralRsProvider;

#[cfg(feature = "provider-mistral")]
pub use mistral::RemoteMistralProvider;

#[cfg(feature = "provider-anthropic")]
pub use anthropic::RemoteAnthropicProvider;

#[cfg(feature = "provider-voyageai")]
pub use voyageai::RemoteVoyageAIProvider;

#[cfg(feature = "provider-cohere")]
pub use cohere::RemoteCohereProvider;

#[cfg(feature = "provider-azure-openai")]
pub use azure_openai::RemoteAzureOpenAIProvider;
