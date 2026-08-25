#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to convert block to a different type")]
    BlockConversion,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("slice conversion error: {0}")]
    SliceConversion(#[from] std::array::TryFromSliceError),

    #[error("invalid data")]
    InvalidData,

    #[error("not enough bytes for {0}")]
    FailedToDecode(String),

    #[error("unexpected end of stream")]
    UnexpectedEndOfStream,

    #[error("buffer overflow - data size exceeds limit")]
    BufferOverflow,

    #[error("insufficient capacity: {0}")]
    InsufficientCapacity(String),

    #[error("invalid image dimensions: width={width}, height={height}")]
    InvalidDimensions { width: usize, height: usize },

    #[error("invalid color depth: {0}")]
    InvalidDepth(u8),

    #[error("invalid stride: {0}")]
    InvalidStride(usize),

    #[error("unsupported ANS data type: {0}-bit types not supported (only 8-bit and 16-bit)")]
    UnsupportedAnsDataType(usize),

    #[error("invalid bit width: {0}")]
    InvalidBitWidth(u8),

    #[error("invalid bitstream alignment")]
    InvalidAlignment,

    #[error("bitstream position overflow")]
    PositionOverflow,

    #[error("invalid Rice parameter: {0}")]
    InvalidRiceParameter(u8),

    #[error("invalid Rice quotient: {0}")]
    RiceQuotientOverflow(usize),

    #[error("invalid symbol count: {0}")]
    InvalidSymbolCount(usize),

    #[error("invalid decoded length: {0}")]
    InvalidDecodedLength(usize),

    #[error("invalid frequency table")]
    InvalidFrequencyTable,

    #[error("allocation exceeds decoder limit: requested={requested}, maximum={maximum}")]
    AllocationLimitExceeded { requested: usize, maximum: usize },
}

pub type Result<T> = std::result::Result<T, Error>;
