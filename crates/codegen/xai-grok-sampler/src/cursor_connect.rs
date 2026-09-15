//! Bounded framing for Cursor's Connect/protobuf streaming transport.
//!
//! Cursor uses the Connect protocol's five-byte frame header: one flags byte
//! followed by a big-endian payload length. The protobuf meaning is handled by
//! the provider adapter; this module only frames bytes and preserves flags.

use thiserror::Error;

/// Maximum individual Connect frame size, matching the pinned Pi reference.
pub const MAX_CONNECT_FRAME_BYTES: usize = 64 * 1024 * 1024;
const CONNECT_FRAME_HEADER_BYTES: usize = 5;
const CONNECT_END_STREAM_FLAG: u8 = 0b0000_0010;

/// One decoded Connect frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectFrame {
    pub flags: u8,
    pub payload: Vec<u8>,
}

impl ConnectFrame {
    /// Whether this frame terminates the Connect stream.
    pub fn is_end_stream(&self) -> bool {
        self.flags & CONNECT_END_STREAM_FLAG != 0
    }
}

/// Errors caused by malformed or over-limit Connect frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum ConnectFrameError {
    #[error("Connect frame exceeds the configured size limit.")]
    FrameTooLarge,
    #[error("Connect stream ended with an incomplete frame.")]
    TruncatedFrame,
}

/// Incremental frame decoder that tolerates arbitrary HTTP/2 chunk boundaries.
#[derive(Debug)]
pub struct ConnectFrameDecoder {
    pending: Vec<u8>,
    max_frame_bytes: usize,
}

impl Default for ConnectFrameDecoder {
    fn default() -> Self {
        Self::with_max_frame_bytes(MAX_CONNECT_FRAME_BYTES)
    }
}

impl ConnectFrameDecoder {
    /// Build a decoder with a caller-specific frame ceiling.
    pub fn with_max_frame_bytes(max_frame_bytes: usize) -> Self {
        Self {
            pending: Vec::new(),
            max_frame_bytes: max_frame_bytes.min(MAX_CONNECT_FRAME_BYTES),
        }
    }

    /// Add response bytes and return every complete frame now available.
    pub fn push(&mut self, mut input: &[u8]) -> Result<Vec<ConnectFrame>, ConnectFrameError> {
        let mut frames = Vec::new();
        while !input.is_empty() {
            if self.pending.len() < CONNECT_FRAME_HEADER_BYTES {
                let needed = CONNECT_FRAME_HEADER_BYTES - self.pending.len();
                let take = needed.min(input.len());
                self.pending.extend_from_slice(&input[..take]);
                input = &input[take..];
                if self.pending.len() < CONNECT_FRAME_HEADER_BYTES {
                    break;
                }
            }

            let payload_len = u32::from_be_bytes([
                self.pending[1],
                self.pending[2],
                self.pending[3],
                self.pending[4],
            ]) as usize;
            if payload_len > self.max_frame_bytes {
                self.pending.clear();
                return Err(ConnectFrameError::FrameTooLarge);
            }

            let expected_len = CONNECT_FRAME_HEADER_BYTES + payload_len;
            let needed = expected_len - self.pending.len();
            let take = needed.min(input.len());
            self.pending.extend_from_slice(&input[..take]);
            input = &input[take..];
            if self.pending.len() < expected_len {
                break;
            }

            let payload = self.pending[CONNECT_FRAME_HEADER_BYTES..].to_vec();
            frames.push(ConnectFrame {
                flags: self.pending[0],
                payload,
            });
            // Do not retain a potentially multi-megabyte allocation after a
            // complete frame has been copied out.
            self.pending = Vec::new();
        }
        Ok(frames)
    }

    /// Assert that the HTTP/2 body ended on a frame boundary.
    pub fn finish(&self) -> Result<(), ConnectFrameError> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err(ConnectFrameError::TruncatedFrame)
        }
    }

    /// Number of bytes retained while waiting for the remainder of a frame.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Inspect the next frame length from retained bytes plus an incoming
    /// slice without copying that slice into the decoder.
    pub fn next_frame_total_len_with(
        &self,
        incoming: &[u8],
    ) -> Result<Option<usize>, ConnectFrameError> {
        let mut header = [0_u8; CONNECT_FRAME_HEADER_BYTES];
        let pending_header_len = self.pending.len().min(CONNECT_FRAME_HEADER_BYTES);
        header[..pending_header_len].copy_from_slice(&self.pending[..pending_header_len]);
        let needed = CONNECT_FRAME_HEADER_BYTES - pending_header_len;
        let take = needed.min(incoming.len());
        header[pending_header_len..pending_header_len + take].copy_from_slice(&incoming[..take]);
        if pending_header_len + take < CONNECT_FRAME_HEADER_BYTES {
            return Ok(None);
        }
        let payload_len = u32::from_be_bytes([header[1], header[2], header[3], header[4]]) as usize;
        if payload_len > self.max_frame_bytes {
            return Err(ConnectFrameError::FrameTooLarge);
        }
        Ok(Some(CONNECT_FRAME_HEADER_BYTES + payload_len))
    }
}

/// Encode one protobuf payload as a Connect frame.
pub fn encode_connect_frame(flags: u8, payload: &[u8]) -> Result<Vec<u8>, ConnectFrameError> {
    if payload.len() > MAX_CONNECT_FRAME_BYTES {
        return Err(ConnectFrameError::FrameTooLarge);
    }
    let length = u32::try_from(payload.len()).map_err(|_| ConnectFrameError::FrameTooLarge)?;
    let mut frame = Vec::with_capacity(CONNECT_FRAME_HEADER_BYTES + payload.len());
    frame.push(flags);
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(payload);
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_and_decodes_flags_and_payload() {
        let encoded =
            encode_connect_frame(CONNECT_END_STREAM_FLAG, b"server done").expect("encode frame");
        let mut decoder = ConnectFrameDecoder::default();
        let frames = decoder.push(&encoded).expect("decode frame");
        assert_eq!(
            frames,
            vec![ConnectFrame {
                flags: CONNECT_END_STREAM_FLAG,
                payload: b"server done".to_vec(),
            }]
        );
        assert!(frames[0].is_end_stream());
        decoder.finish().expect("complete stream");
    }

    #[test]
    fn incremental_decoder_handles_fragmented_and_coalesced_frames() {
        let first = encode_connect_frame(0, b"first").expect("first frame");
        let second = encode_connect_frame(0, b"second").expect("second frame");
        let combined = [first, second].concat();
        let mut decoder = ConnectFrameDecoder::default();
        let mut frames = Vec::new();
        for byte in combined.chunks(1) {
            frames.extend(decoder.push(byte).expect("decode byte fragment"));
        }
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].payload, b"first");
        assert_eq!(frames[1].payload, b"second");
        decoder.finish().expect("complete stream");
    }

    #[test]
    fn rejects_frame_length_over_limit_before_allocating_payload() {
        let oversized_len = (MAX_CONNECT_FRAME_BYTES as u32 + 1).to_be_bytes();
        let header = [
            0,
            oversized_len[0],
            oversized_len[1],
            oversized_len[2],
            oversized_len[3],
        ];
        let mut decoder = ConnectFrameDecoder::default();
        assert_eq!(decoder.push(&header), Err(ConnectFrameError::FrameTooLarge));
    }

    #[test]
    fn cursor_decoder_honors_a_smaller_caller_frame_limit() {
        let mut decoder = ConnectFrameDecoder::with_max_frame_bytes(3);
        let frame = encode_connect_frame(0, b"four").expect("global frame size");
        assert_eq!(decoder.push(&frame), Err(ConnectFrameError::FrameTooLarge));
    }

    #[test]
    fn reports_stream_ending_mid_frame() {
        let encoded = encode_connect_frame(0, b"body").expect("frame");
        let mut decoder = ConnectFrameDecoder::default();
        assert!(
            decoder
                .push(&encoded[..7])
                .expect("partial frame")
                .is_empty()
        );
        assert_eq!(decoder.finish(), Err(ConnectFrameError::TruncatedFrame));
    }
}
