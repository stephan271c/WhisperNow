//! llama.cpp, linked at run time from the shared libraries of a pinned release (b11259).
//!
//! Only its public C API is used. The libraries are loaded from one directory, once per
//! process, and stay loaded: ggml's backends do not support being unloaded.
#![allow(unsafe_code)]

mod ffi;

use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    path::{Path, PathBuf},
    ptr::NonNull,
    rc::Rc,
    sync::{Mutex, PoisonError},
};

use libloading::Library;
use thiserror::Error;
use whispernow_core::llm::local::{DecodeContext, LocalDecodeError, Outputs, Token};

/// The release's libraries, in load order: each needs the ones before it.
#[cfg(target_os = "linux")]
const LIBRARIES: [&str; 3] = ["libggml-base.so.0", "libggml.so.0", "libllama.so.0"];
#[cfg(target_os = "macos")]
const LIBRARIES: [&str; 3] = [
    "libggml-base.0.dylib",
    "libggml.0.dylib",
    "libllama.0.dylib",
];
#[cfg(target_os = "windows")]
const LIBRARIES: [&str; 3] = ["ggml-base.dll", "ggml.dll", "llama.dll"];
/// Loaded first when present: the Windows CPU backends need LLVM's OpenMP runtime, which
/// ships beside them.
#[cfg(target_os = "windows")]
const PRELOAD: &[&str] = &["libomp.dll"];
#[cfg(not(target_os = "windows"))]
const PRELOAD: &[&str] = &[];

/// Layers to offload to a GPU: all of them, as the training side serves.
const ALL_LAYERS: i32 = 99;

/// The CUDA pack's libraries: NVIDIA's CUDA 13 runtime, in load order, then the CUDA backend
/// of the same llama.cpp release's CUDA 13.4 build. The backend is added beside the bundled
/// ones, so a GPU CUDA cannot run on still has Vulkan.
#[derive(Clone, Copy, Debug)]
pub struct CudaLibraries {
    pub runtime: [&'static str; 3],
    pub backend: &'static str,
}

impl CudaLibraries {
    pub fn files(&self) -> impl Iterator<Item = &'static str> {
        self.runtime.into_iter().chain([self.backend])
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(crate) const CUDA_PACK_LIBRARIES: CudaLibraries = CudaLibraries {
    runtime: ["libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13"],
    backend: "libggml-cuda.so",
};
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub(crate) const CUDA_PACK_LIBRARIES: CudaLibraries = CudaLibraries {
    runtime: ["cudart64_13.dll", "cublasLt64_13.dll", "cublas64_13.dll"],
    backend: "ggml-cuda.dll",
};
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86_64")
))]
pub const CUDA_LIBRARIES: Option<CudaLibraries> = Some(CUDA_PACK_LIBRARIES);
#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86_64")
)))]
pub const CUDA_LIBRARIES: Option<CudaLibraries> = None;

/// The oldest NVIDIA architecture the CUDA 13 build has code for (Turing, compute 7.5). On
/// an older GPU its first kernel would fail, which aborts the process, so CUDA skips it.
const CUDA_MIN_COMPUTE: (c_int, c_int) = (7, 5);
/// `cudaDevAttrComputeCapabilityMajor` and `...Minor` of `enum cudaDeviceAttr`.
const CUDA_COMPUTE_MAJOR: c_int = 75;
const CUDA_COMPUTE_MINOR: c_int = 76;

/// `cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)`.
type CudaDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;

/// The CUDA pack's device query, once the pack is loaded.
static CUDA: Mutex<Option<CudaDeviceGetAttribute>> = Mutex::new(None);
/// Held while llama.cpp loads, so its global state is initialized once.
static LOADED: Mutex<Option<&'static Llama>> = Mutex::new(None);

#[derive(Debug, Error)]
pub(crate) enum LlamaError {
    #[error("llama.cpp could not be loaded from {directory}: {message}")]
    Library { directory: PathBuf, message: String },
    #[error("the model is not installed")]
    NotInstalled,
    #[error("the model could not be loaded")]
    Model,
    #[error("the model's context could not be created")]
    Context,
    #[error("the model's memory test decode failed (status {0})")]
    Probe(i32),
}

pub struct Llama {
    api: ffi::Api,
    _libraries: Vec<Library>,
}

// SAFETY: the function table holds plain function pointers into libraries that are never
// unloaded; llama.cpp's global state is initialized once, under `LOADED`'s lock.
unsafe impl Send for Llama {}
// SAFETY: as above; the functions used without a model or context are thread-safe.
unsafe impl Sync for Llama {}

impl Llama {
    /// Loads llama.cpp from `directory`, or returns the copy already loaded.
    pub(crate) fn load(directory: &Path) -> Result<&'static Self, LlamaError> {
        let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(llama) = *loaded {
            return Ok(llama);
        }
        let failed = |message: String| LlamaError::Library {
            directory: directory.to_owned(),
            message,
        };
        let mut libraries = Vec::new();
        for name in PRELOAD {
            let path = directory.join(name);
            if path.is_file() {
                libraries.push(open(&path).map_err(|error| failed(error.to_string()))?);
            }
        }
        let mut core = Vec::with_capacity(LIBRARIES.len());
        for name in LIBRARIES {
            core.push(open(&directory.join(name)).map_err(|error| failed(error.to_string()))?);
        }
        // SAFETY: these are the pinned release's libraries, and `Llama` keeps them loaded
        // for the life of the process.
        let api = unsafe { ffi::Api::resolve(&core[0], &core[1], &core[2]) }
            .map_err(|error| failed(error.to_string()))?;
        libraries.extend(core);
        // SAFETY: the callback is a plain function that never unwinds.
        unsafe {
            (api.ggml_log_set)(Some(log), std::ptr::null_mut());
            (api.llama_log_set)(Some(log), std::ptr::null_mut());
            if (api.ggml_backend_reg_count)() == 0 {
                let directory = CString::new(directory.as_os_str().as_encoded_bytes())
                    .map_err(|error| failed(error.to_string()))?;
                (api.ggml_backend_load_all_from_path)(directory.as_ptr());
            }
            (api.llama_backend_init)();
        }
        let llama: &'static Self = Box::leak(Box::new(Self {
            api,
            _libraries: libraries,
        }));
        *loaded = Some(llama);
        Ok(llama)
    }

    /// Adds the CUDA backend from the CUDA pack in `directory`, once per process.
    pub(crate) fn load_cuda(&self, directory: &Path) -> Result<(), LlamaError> {
        let failed = |message: String| LlamaError::Library {
            directory: directory.to_owned(),
            message,
        };
        let names =
            CUDA_LIBRARIES.ok_or_else(|| failed("CUDA is not supported here".to_owned()))?;
        let mut cuda = CUDA.lock().unwrap_or_else(PoisonError::into_inner);
        if cuda.is_some() {
            return Ok(());
        }
        let mut libraries = Vec::with_capacity(names.runtime.len() + 1);
        for name in names.runtime {
            libraries.push(open(&directory.join(name)).map_err(|error| failed(error.to_string()))?);
        }
        // SAFETY: the CUDA runtime's documented signature; the library is kept loaded below.
        let device_attribute = unsafe {
            *libraries[0]
                .get::<CudaDeviceGetAttribute>(b"cudaDeviceGetAttribute\0")
                .map_err(|error| failed(error.to_string()))?
        };
        // Opened here first, so that on Windows its dependencies resolve from its own
        // directory; ggml's own load below then finds it loaded.
        let backend = directory.join(names.backend);
        libraries.push(open(&backend).map_err(|error| failed(error.to_string()))?);
        let path = CString::new(backend.as_os_str().as_encoded_bytes())
            .map_err(|error| failed(error.to_string()))?;
        // SAFETY: a NUL-terminated path to a backend of the same release.
        if unsafe { (self.api.ggml_backend_load)(path.as_ptr()) }.is_null() {
            return Err(failed("ggml did not load the CUDA backend".to_owned()));
        }
        // ggml now holds the backend; it and its runtime must never be unloaded.
        std::mem::forget(libraries);
        *cuda = Some(device_attribute);
        Ok(())
    }

    /// Whether the CUDA pack is loaded in this process. Its files stay in use until the
    /// process exits.
    #[must_use]
    pub fn cuda_loaded() -> bool {
        CUDA.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    /// The GPUs to try, best first: CUDA devices, then other discrete GPUs, then integrated
    /// ones; within each, the most memory first. With the CUDA pack loaded the same GPU is
    /// also listed under Vulkan, which is where it falls back to.
    fn gpus(&self) -> Vec<(ffi::Device, String)> {
        let api = &self.api;
        let device_attribute = *CUDA.lock().unwrap_or_else(PoisonError::into_inner);
        let mut devices: Vec<(u8, usize, ffi::Device, String)> = Vec::new();
        // SAFETY: device handles stay valid while the backends are loaded.
        let count = unsafe { (api.ggml_backend_dev_count)() };
        for index in 0..count {
            // SAFETY: device handles stay valid while the backends are loaded.
            let device = unsafe { (api.ggml_backend_dev_get)(index) };
            if device.is_null() {
                continue;
            }
            // SAFETY: device handles stay valid while the backends are loaded.
            let (name, kind) = unsafe {
                (
                    text((api.ggml_backend_dev_name)(device)),
                    (api.ggml_backend_dev_type)(device),
                )
            };
            let rank = match name.strip_prefix("CUDA").map(str::parse::<c_int>) {
                Some(Ok(ordinal)) => device_attribute
                    .is_none_or(|attribute| cuda_runs_on(attribute, ordinal, &name))
                    .then_some(0),
                _ => match kind {
                    ffi::DEVICE_TYPE_GPU => Some(1),
                    ffi::DEVICE_TYPE_IGPU => Some(2),
                    _ => None,
                },
            };
            let Some(rank) = rank else {
                continue;
            };
            let (mut free, mut total) = (0, 0);
            // SAFETY: device handles stay valid while the backends are loaded.
            let description = unsafe {
                (api.ggml_backend_dev_memory)(device, &raw mut free, &raw mut total);
                text((api.ggml_backend_dev_description)(device))
            };
            devices.push((rank, total, device, format!("{name} ({description})")));
        }
        devices.sort_by_key(|&(rank, memory, ..)| (rank, std::cmp::Reverse(memory)));
        devices
            .into_iter()
            .map(|(_, _, device, description)| (device, description))
            .collect()
    }
}

/// Whether the CUDA build has code for the GPU at CUDA `ordinal`, as ggml numbers them.
fn cuda_runs_on(device_attribute: CudaDeviceGetAttribute, ordinal: c_int, name: &str) -> bool {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: plain out-parameters; the runtime stays loaded.
    let known = unsafe {
        device_attribute(&raw mut major, CUDA_COMPUTE_MAJOR, ordinal) == 0
            && device_attribute(&raw mut minor, CUDA_COMPUTE_MINOR, ordinal) == 0
    };
    let runs = known && (major, minor) >= CUDA_MIN_COMPUTE;
    if !runs {
        tracing::info!(target: "whispernow::llama", device = name, major, minor, "CUDA has no code for this GPU; it is left to the other backends");
    }
    runs
}

fn open(path: &Path) -> Result<Library, libloading::Error> {
    // SAFETY: loading runs the library's initializers; these are llama.cpp's own.
    unsafe {
        #[cfg(target_os = "windows")]
        {
            // Dependencies resolve from the library's own directory.
            libloading::os::windows::Library::load_with_flags(
                path,
                libloading::os::windows::LOAD_WITH_ALTERED_SEARCH_PATH,
            )
            .map(Library::from)
        }
        #[cfg(not(target_os = "windows"))]
        {
            Library::new(path)
        }
    }
}

/// # Safety
/// `pointer` must be NULL or a NUL-terminated string.
unsafe fn text(pointer: *const c_char) -> String {
    if pointer.is_null() {
        String::new()
    } else {
        // SAFETY: guaranteed by the caller.
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

unsafe extern "C" fn log(level: c_int, message: *const c_char, _data: *mut c_void) {
    if level == ffi::LOG_LEVEL_ERROR {
        // SAFETY: llama.cpp passes a NUL-terminated message.
        let message = unsafe { text(message) };
        tracing::warn!(target: "whispernow::llama", message = message.trim_end(), "llama.cpp error");
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Backend {
    /// A GPU, by its backend's name and description.
    Gpu(String),
    Cpu,
}

pub(crate) struct LlamaModel {
    llama: &'static Llama,
    model: NonNull<ffi::Model>,
    vocab: *const ffi::Vocab,
    backend: Backend,
}

impl LlamaModel {
    /// Loads a GGUF on the first GPU that takes it, else on the CPU.
    pub(crate) fn load(llama: &'static Llama, path: &Path) -> Result<Self, LlamaError> {
        let path =
            CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| LlamaError::Model)?;
        for (device, description) in llama.gpus() {
            let mut devices = [device, std::ptr::null_mut()];
            match Self::load_on(llama, &path, devices.as_mut_ptr(), ALL_LAYERS) {
                Ok((model, vocab)) => {
                    return Ok(Self {
                        llama,
                        model,
                        vocab,
                        backend: Backend::Gpu(description),
                    });
                }
                Err(_) => {
                    tracing::warn!(target: "whispernow::llama", device = description, "the GPU did not take the model; trying the next device");
                }
            }
        }
        // An empty device list keeps everything, the KV cache included, on the CPU.
        let mut devices = [std::ptr::null_mut()];
        let (model, vocab) = Self::load_on(llama, &path, devices.as_mut_ptr(), 0)?;
        Ok(Self {
            llama,
            model,
            vocab,
            backend: Backend::Cpu,
        })
    }

    fn load_on(
        llama: &Llama,
        path: &CStr,
        devices: *mut ffi::Device,
        gpu_layers: i32,
    ) -> Result<(NonNull<ffi::Model>, *const ffi::Vocab), LlamaError> {
        let api = &llama.api;
        // SAFETY: the parameters are the library's defaults with valid pointers; `devices`
        // is NULL-terminated and only read during the load.
        unsafe {
            let mut params = (api.llama_model_default_params)();
            params.devices = devices;
            params.n_gpu_layers = gpu_layers;
            let model = NonNull::new((api.llama_model_load_from_file)(path.as_ptr(), params))
                .ok_or(LlamaError::Model)?;
            let vocab = (api.llama_model_get_vocab)(model.as_ptr());
            if vocab.is_null() {
                (api.llama_model_free)(model.as_ptr());
                return Err(LlamaError::Model);
            }
            Ok((model, vocab))
        }
    }

    #[must_use]
    pub(crate) fn backend(&self) -> &Backend {
        &self.backend
    }

    /// The BOS token as text, for the chat template; empty when the model has none.
    #[must_use]
    pub(crate) fn bos_text(&self) -> String {
        // SAFETY: the vocabulary lives as long as the model.
        let bos = unsafe { (self.llama.api.llama_vocab_bos)(self.vocab) };
        if bos < 0 {
            return String::new();
        }
        String::from_utf8_lossy(&self.token_text(bos, true)).into_owned()
    }

    fn token_text(&self, token: Token, special: bool) -> Vec<u8> {
        let mut buffer = vec![0_u8; 32];
        loop {
            let length = i32::try_from(buffer.len()).unwrap_or(i32::MAX);
            // SAFETY: the buffer holds `length` bytes.
            let written = unsafe {
                (self.llama.api.llama_token_to_piece)(
                    self.vocab,
                    token,
                    buffer.as_mut_ptr().cast(),
                    length,
                    0,
                    special,
                )
            };
            match usize::try_from(written) {
                Ok(written) => {
                    buffer.truncate(written);
                    return buffer;
                }
                Err(_) => buffer.resize(written.unsigned_abs() as usize, 0),
            }
        }
    }

    pub(crate) fn tokenize(
        &self,
        text: &[u8],
        parse_special: bool,
    ) -> Result<Vec<Token>, LocalDecodeError> {
        let length = i32::try_from(text.len()).map_err(|_| LocalDecodeError::Tokenize)?;
        let mut tokens: Vec<Token> = vec![0; text.len() + 2];
        loop {
            let capacity = i32::try_from(tokens.len()).map_err(|_| LocalDecodeError::Tokenize)?;
            // SAFETY: `text` holds `length` bytes and `tokens` `capacity` tokens.
            let count = unsafe {
                (self.llama.api.llama_tokenize)(
                    self.vocab,
                    text.as_ptr().cast(),
                    length,
                    tokens.as_mut_ptr(),
                    capacity,
                    false,
                    parse_special,
                )
            };
            if count == i32::MIN {
                return Err(LocalDecodeError::Tokenize);
            }
            match usize::try_from(count) {
                Ok(count) => {
                    tokens.truncate(count);
                    return Ok(tokens);
                }
                Err(_) => tokens.resize(count.unsigned_abs() as usize, 0),
            }
        }
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        // SAFETY: sessions hold the model, so none outlives it.
        unsafe { (self.llama.api.llama_model_free)(self.model.as_ptr()) };
    }
}

/// The tokens a speculative pass verifies at most, the sampled one included.
const SPECULATIVE_OUTPUTS: u32 = 5;
const _: () =
    assert!(SPECULATIVE_OUTPUTS as usize == 1 + whispernow_core::llm::local::DRAFT_TOKENS);

/// One context on a model: its memory holds one sequence.
pub(crate) struct LlamaContext {
    model: Rc<LlamaModel>,
    context: NonNull<ffi::Context>,
    memory: *mut ffi::Memory,
    batch: ffi::Batch,
    vocab_size: usize,
    context_size: usize,
    batch_size: usize,
    micro_batch_size: usize,
    checkpoints: bool,
}

impl LlamaContext {
    /// Creates a context of at least `context_size` tokens as `llama-server` b11259 does for
    /// one slot with speculative decoding: batches of 2,048 tokens in micro-batches of 512,
    /// room for 4 draft tokens' outputs, recurrent state that can roll back 4 tokens
    /// (`n_rs_seq`) as the patched server sets up for `ngram-simple`, Flash Attention when
    /// the backend supports it, and a warm-up decode.
    pub(crate) fn new(
        model: Rc<LlamaModel>,
        context_size: u32,
        threads: i32,
    ) -> Result<Self, LlamaError> {
        let api = &model.llama.api;
        // SAFETY: the parameters are the library's defaults with plain values set.
        let context = unsafe {
            let mut params = (api.llama_context_default_params)();
            params.n_ctx = context_size;
            params.n_batch = 2_048;
            params.n_ubatch = 512;
            params.n_seq_max = 1;
            params.n_rs_seq = SPECULATIVE_OUTPUTS - 1;
            params.n_outputs_max = SPECULATIVE_OUTPUTS;
            params.n_outputs_max_per_seq = SPECULATIVE_OUTPUTS;
            params.n_threads = threads;
            params.n_threads_batch = threads;
            // The server's defaults where the library's differ.
            params.swa_full = false;
            params.no_perf = true;
            NonNull::new((api.llama_init_from_model)(model.model.as_ptr(), params))
                .ok_or(LlamaError::Context)?
        };
        // SAFETY: the context is valid until `Drop`.
        let (memory, context_size, batch_size, micro_batch_size, rollback, vocab_size) = unsafe {
            (
                (api.llama_get_memory)(context.as_ptr()),
                (api.llama_n_ctx)(context.as_ptr()),
                (api.llama_n_batch)(context.as_ptr()),
                (api.llama_n_ubatch)(context.as_ptr()),
                (api.llama_n_rs_seq)(context.as_ptr()),
                (api.llama_vocab_n_tokens)(model.vocab),
            )
        };
        let batch_size = batch_size.clamp(1, i32::MAX as u32);
        // SAFETY: allocates token, position, sequence and output arrays of `batch_size`.
        let batch = unsafe { (api.llama_batch_init)(batch_size.cast_signed(), 0, 1) };
        let batch_size = batch_size as usize;
        // SAFETY: the model is loaded.
        let n_swa = unsafe { (api.llama_model_n_swa)(model.model.as_ptr()) };
        let mut context = Self {
            model,
            context,
            memory,
            batch,
            vocab_size: usize::try_from(vocab_size).unwrap_or_default(),
            context_size: context_size as usize,
            batch_size,
            micro_batch_size: micro_batch_size as usize,
            checkpoints: true,
        };
        if memory.is_null() {
            return Err(LlamaError::Context);
        }
        let tail_removal_failed = context.probe_rollback()?;
        context.checkpoints = rollback > 0 || tail_removal_failed || n_swa > 0;
        Ok(context)
    }

    /// Whether the memory cannot remove a sequence's tail, by trying it after two tokens
    /// (llama.cpp's `common_context_can_seq_rm`). Doubles as the server's warm-up decode.
    fn probe_rollback(&mut self) -> Result<bool, LlamaError> {
        let api = &self.model.llama.api;
        self.clear();
        self.fill(&[0, 0], 0, Outputs::Last);
        // SAFETY: the batch holds two tokens of sequence 0.
        let status = unsafe { (api.llama_decode)(self.context.as_ptr(), self.batch) };
        if status != 0 {
            self.clear();
            return Err(LlamaError::Probe(status));
        }
        // SAFETY: the memory belongs to the live context.
        let removed = unsafe { (api.llama_memory_seq_rm)(self.memory, 0, 1, -1) };
        self.clear();
        // SAFETY: as above.
        unsafe { (api.llama_synchronize)(self.context.as_ptr()) };
        Ok(!removed)
    }

    fn fill(&mut self, tokens: &[Token], first_position: usize, outputs: Outputs) {
        let count = tokens.len().min(self.batch_size);
        // SAFETY: `llama_batch_init` allocated `batch_size` entries in each array, each with
        // room for one sequence id.
        unsafe {
            for (index, &token) in tokens[..count].iter().enumerate() {
                *self.batch.token.add(index) = token;
                *self.batch.pos.add(index) =
                    i32::try_from(first_position + index).unwrap_or(i32::MAX);
                *self.batch.n_seq_id.add(index) = 1;
                *(*self.batch.seq_id.add(index)) = 0;
                *self.batch.logits.add(index) = i8::from(match outputs {
                    Outputs::None => false,
                    Outputs::Last => index + 1 == count,
                    Outputs::All => true,
                });
            }
        }
        self.batch.n_tokens = i32::try_from(count).unwrap_or(i32::MAX);
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        let api = &self.model.llama.api;
        // SAFETY: both were created in `new` and are freed once.
        unsafe {
            (api.llama_batch_free)(self.batch);
            (api.llama_free)(self.context.as_ptr());
        }
    }
}

impl DecodeContext for LlamaContext {
    fn tokenize(&self, text: &[u8], parse_special: bool) -> Result<Vec<Token>, LocalDecodeError> {
        self.model.tokenize(text, parse_special)
    }

    fn piece(&self, token: Token) -> Vec<u8> {
        self.model.token_text(token, false)
    }

    fn is_end_of_generation(&self, token: Token) -> bool {
        // SAFETY: the vocabulary lives as long as the model.
        unsafe { (self.model.llama.api.llama_vocab_is_eog)(self.model.vocab, token) }
    }

    fn vocabulary_size(&self) -> usize {
        self.vocab_size
    }

    fn suppressed_tokens(&self) -> Vec<Token> {
        let mut count = 0;
        // SAFETY: returns `count` tokens owned by the vocabulary, or NULL.
        unsafe {
            let tokens = (self.model.llama.api.llama_vocab_get_suppress_tokens)(
                self.model.vocab,
                &raw mut count,
            );
            match usize::try_from(count) {
                Ok(count) if count > 0 && !tokens.is_null() => {
                    std::slice::from_raw_parts(tokens, count).to_vec()
                }
                _ => Vec::new(),
            }
        }
    }

    fn context_size(&self) -> usize {
        self.context_size
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn micro_batch_size(&self) -> usize {
        self.micro_batch_size
    }

    fn checkpoints_prompt(&self) -> bool {
        self.checkpoints
    }

    fn clear(&mut self) {
        // SAFETY: the memory belongs to the live context.
        unsafe { (self.model.llama.api.llama_memory_clear)(self.memory, true) };
    }

    fn decode(
        &mut self,
        tokens: &[Token],
        first_position: usize,
        outputs: Outputs,
    ) -> Result<(), LocalDecodeError> {
        if tokens.is_empty() || tokens.len() > self.batch_size {
            return Err(LocalDecodeError::Decode(-1));
        }
        self.fill(tokens, first_position, outputs);
        // SAFETY: the batch was filled above for sequence 0.
        let status =
            unsafe { (self.model.llama.api.llama_decode)(self.context.as_ptr(), self.batch) };
        if status == 0 {
            Ok(())
        } else {
            Err(LocalDecodeError::Decode(status))
        }
    }

    fn logits(&mut self, batch_index: usize) -> Result<&[f32], LocalDecodeError> {
        let index = i32::try_from(batch_index).map_err(|_| LocalDecodeError::Logits)?;
        // SAFETY: returns `n_vocab` logits owned by the context until its next decode, or
        // NULL when that position produced none.
        unsafe {
            let logits = (self.model.llama.api.llama_get_logits_ith)(self.context.as_ptr(), index);
            if logits.is_null() {
                Err(LocalDecodeError::Logits)
            } else {
                Ok(std::slice::from_raw_parts(logits, self.vocab_size))
            }
        }
    }

    fn remove_from(&mut self, from: usize) -> Result<(), LocalDecodeError> {
        let from = i32::try_from(from).map_err(|_| LocalDecodeError::Memory)?;
        // SAFETY: the memory belongs to the live context.
        if unsafe { (self.model.llama.api.llama_memory_seq_rm)(self.memory, 0, from, -1) } {
            Ok(())
        } else {
            Err(LocalDecodeError::Memory)
        }
    }
}
