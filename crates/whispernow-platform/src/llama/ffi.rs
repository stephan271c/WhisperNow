//! The parts of the pinned llama.cpp release's public C API (`include/llama.h`,
//! `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`) the cleanup engine uses, resolved at
//! run time.
//!
//! The structs mirror the C layouts of that release exactly; the size checks below were
//! taken from the headers with `offsetof` and `sizeof`. Bumping the pinned release means
//! re-checking every field against its headers.

use std::ffi::{c_char, c_int, c_void};

use libloading::Library;

pub(super) type Token = i32;

#[repr(C)]
pub(super) struct Opaque {
    _private: [u8; 0],
}

pub(super) type Model = Opaque;
pub(super) type Context = Opaque;
pub(super) type Vocab = Opaque;
pub(super) type Memory = Opaque;
pub(super) type Device = *mut Opaque;

pub(super) type LogCallback =
    unsafe extern "C" fn(level: c_int, text: *const c_char, data: *mut c_void);
pub(super) type AbortCallback = unsafe extern "C" fn(data: *mut c_void) -> bool;

/// `enum ggml_backend_dev_type`.
pub(super) const DEVICE_TYPE_GPU: c_int = 1;
pub(super) const DEVICE_TYPE_IGPU: c_int = 2;
/// `enum ggml_log_level`.
pub(super) const LOG_LEVEL_ERROR: c_int = 4;

/// `struct llama_model_params`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ModelParams {
    /// NULL-terminated list of devices; NULL uses every available one.
    pub devices: *mut Device,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: c_int,
    pub load_mode: c_int,
    pub lazy_mode: c_int,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: *const c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

/// `struct llama_context_params`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: c_int,
    pub rope_scaling_type: c_int,
    pub pooling_type: c_int,
    pub attention_type: c_int,
    pub flash_attn_type: c_int,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: *const c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: c_int,
    pub type_v: c_int,
    pub abort_callback: Option<AbortCallback>,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
    pub ctx_other: *mut c_void,
}

/// `struct llama_batch`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Batch {
    pub n_tokens: i32,
    pub token: *mut Token,
    pub embd: *mut f32,
    pub pos: *mut i32,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32,
    pub logits: *mut i8,
}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<ModelParams>() == 80);
    assert!(std::mem::offset_of!(ModelParams, n_gpu_layers) == 16);
    assert!(std::mem::offset_of!(ModelParams, tensor_split) == 40);
    assert!(std::mem::offset_of!(ModelParams, vocab_only) == 72);
    assert!(std::mem::offset_of!(ModelParams, load_mtp) == 77);
    assert!(size_of::<ContextParams>() == 160);
    assert!(std::mem::offset_of!(ContextParams, n_rs_seq) == 16);
    assert!(std::mem::offset_of!(ContextParams, n_threads) == 28);
    assert!(std::mem::offset_of!(ContextParams, flash_attn_type) == 52);
    assert!(std::mem::offset_of!(ContextParams, yarn_orig_ctx) == 80);
    assert!(std::mem::offset_of!(ContextParams, cb_eval) == 88);
    assert!(std::mem::offset_of!(ContextParams, type_k) == 104);
    assert!(std::mem::offset_of!(ContextParams, abort_callback) == 112);
    assert!(std::mem::offset_of!(ContextParams, embeddings) == 128);
    assert!(std::mem::offset_of!(ContextParams, kv_unified) == 133);
    assert!(std::mem::offset_of!(ContextParams, samplers) == 136);
    assert!(std::mem::offset_of!(ContextParams, ctx_other) == 152);
    assert!(size_of::<Batch>() == 56);
};

macro_rules! api {
    ($($library:ident { $($name:ident: fn($($argument:ty),*) $(-> $output:ty)?;)* })*) => {
        /// The resolved functions. Valid while the libraries they came from stay loaded,
        /// which is for the life of the process.
        #[allow(clippy::struct_field_names)]
        pub(super) struct Api {
            $($(pub $name: unsafe extern "C" fn($($argument),*) $(-> $output)?,)*)*
        }

        impl Api {
            /// # Safety
            /// The libraries must be the pinned llama.cpp release's, so each symbol has the
            /// declared signature, and must stay loaded while the table is used.
            pub(super) unsafe fn resolve(
                $($library: &Library),*
            ) -> Result<Self, libloading::Error> {
                Ok(Self {
                    $($(
                        // SAFETY: the caller guarantees the signature and the lifetime.
                        $name: unsafe {
                            *$library.get::<unsafe extern "C" fn($($argument),*) $(-> $output)?>(
                                concat!(stringify!($name), "\0").as_bytes(),
                            )?
                        },
                    )*)*
                })
            }
        }
    };
}

api! {
    ggml_base {
        ggml_log_set: fn(Option<LogCallback>, *mut c_void);
        ggml_backend_dev_description: fn(Device) -> *const c_char;
        ggml_backend_dev_memory: fn(Device, *mut usize, *mut usize);
        ggml_backend_dev_type: fn(Device) -> c_int;
    }
    ggml {
        ggml_backend_load_all_from_path: fn(*const c_char);
        ggml_backend_reg_count: fn() -> usize;
        ggml_backend_dev_count: fn() -> usize;
        ggml_backend_dev_get: fn(usize) -> Device;
    }
    llama {
        llama_backend_init: fn();
        llama_log_set: fn(Option<LogCallback>, *mut c_void);
        llama_model_default_params: fn() -> ModelParams;
        llama_model_load_from_file: fn(*const c_char, ModelParams) -> *mut Model;
        llama_model_free: fn(*mut Model);
        llama_model_get_vocab: fn(*const Model) -> *const Vocab;
        llama_model_n_swa: fn(*const Model) -> i32;
        llama_vocab_n_tokens: fn(*const Vocab) -> i32;
        llama_vocab_is_eog: fn(*const Vocab, Token) -> bool;
        llama_vocab_bos: fn(*const Vocab) -> Token;
        llama_vocab_get_suppress_tokens: fn(*const Vocab, *mut i32) -> *const Token;
        llama_context_default_params: fn() -> ContextParams;
        llama_init_from_model: fn(*mut Model, ContextParams) -> *mut Context;
        llama_free: fn(*mut Context);
        llama_n_ctx: fn(*const Context) -> u32;
        llama_n_batch: fn(*const Context) -> u32;
        llama_n_ubatch: fn(*const Context) -> u32;
        llama_n_rs_seq: fn(*const Context) -> u32;
        llama_get_memory: fn(*const Context) -> *mut Memory;
        llama_memory_clear: fn(*mut Memory, bool);
        llama_memory_seq_rm: fn(*mut Memory, i32, i32, i32) -> bool;
        llama_tokenize: fn(*const Vocab, *const c_char, i32, *mut Token, i32, bool, bool) -> i32;
        llama_token_to_piece: fn(*const Vocab, Token, *mut c_char, i32, i32, bool) -> i32;
        llama_batch_init: fn(i32, i32, i32) -> Batch;
        llama_batch_free: fn(Batch);
        llama_decode: fn(*mut Context, Batch) -> i32;
        llama_get_logits_ith: fn(*mut Context, i32) -> *mut f32;
        llama_synchronize: fn(*mut Context);
    }
}
