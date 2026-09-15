//! Handwritten wire codec for Cursor's `aiserver.v1.AvailableModels` RPC.
//!
//! The pinned `agent.v1` schema covers the agent service. Cursor defines this
//! second catalog RPC in a separate, much larger schema; this module keeps only
//! the fields needed to merge parameterized model metadata.

use thiserror::Error;

const MAX_MODELS: usize = 2_048;
const MAX_VARIANTS_PER_MODEL: usize = 128;
const MAX_PARAMETERS_PER_VARIANT: usize = 32;

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CursorModelParameter {
    pub id: String,
    pub value: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CursorParameterizedVariant {
    pub parameters: Vec<CursorModelParameter>,
    pub is_max_mode: bool,
    pub is_default_max_config: Option<bool>,
    pub is_default_non_max_config: Option<bool>,
    pub display_name: Option<String>,
    pub display_name_outside_picker: Option<String>,
    pub variant_string_representation: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CursorParameterizedModel {
    pub name: String,
    pub client_display_name: Option<String>,
    pub server_model_name: Option<String>,
    pub supports_max_mode: Option<bool>,
    pub supports_non_max_mode: Option<bool>,
    pub supports_images: Option<bool>,
    pub context_token_limit: Option<u64>,
    pub context_token_limit_for_max_mode: Option<u64>,
    pub variants: Vec<CursorParameterizedVariant>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum CursorWireError {
    #[error("Cursor model catalog contains malformed protobuf data.")]
    Malformed,
    #[error("Cursor model catalog exceeds a supported item limit.")]
    TooManyItems,
    #[error("Cursor model catalog contains invalid UTF-8 text.")]
    InvalidText,
}

/// Encode `AvailableModelsRequest { use_model_parameters: true, do_not_use_markdown: true }`.
pub fn encode_available_models_request() -> Vec<u8> {
    let mut out = Vec::with_capacity(4);
    encode_varint((5 << 3) | 0, &mut out);
    encode_varint(1, &mut out);
    encode_varint((7 << 3) | 0, &mut out);
    encode_varint(1, &mut out);
    out
}

/// Decode `AvailableModelsResponse.models` and the model/variant fields Pi consumes.
pub fn decode_available_models_response(
    bytes: &[u8],
) -> Result<Vec<CursorParameterizedModel>, CursorWireError> {
    let mut reader = Reader::new(bytes);
    let mut models = Vec::new();
    while !reader.is_empty() {
        let (field, wire_type) = reader.read_key()?;
        if field == 2 && wire_type == 2 {
            if models.len() >= MAX_MODELS {
                return Err(CursorWireError::TooManyItems);
            }
            let payload = reader.read_bytes()?;
            let model = decode_parameterized_model(payload)?;
            if !model.name.is_empty() {
                models.push(model);
            }
        } else {
            reader.skip(wire_type)?;
        }
    }
    Ok(models)
}

fn decode_parameterized_model(bytes: &[u8]) -> Result<CursorParameterizedModel, CursorWireError> {
    let mut reader = Reader::new(bytes);
    let mut model = CursorParameterizedModel::default();
    while !reader.is_empty() {
        let (field, wire_type) = reader.read_key()?;
        match (field, wire_type) {
            (1, 2) => model.name = reader.read_string()?,
            (10, 0) => model.supports_images = Some(reader.read_varint()? != 0),
            (14, 0) => model.supports_max_mode = Some(reader.read_varint()? != 0),
            (15, 0) => model.context_token_limit = Some(reader.read_varint()?),
            (16, 0) => model.context_token_limit_for_max_mode = Some(reader.read_varint()?),
            (17, 2) => model.client_display_name = Some(reader.read_string()?),
            (18, 2) => model.server_model_name = Some(reader.read_string()?),
            (19, 0) => model.supports_non_max_mode = Some(reader.read_varint()? != 0),
            (30, 2) => {
                if model.variants.len() >= MAX_VARIANTS_PER_MODEL {
                    return Err(CursorWireError::TooManyItems);
                }
                model
                    .variants
                    .push(decode_parameterized_variant(reader.read_bytes()?)?);
            }
            (_, expected_wire_type) if expected_wire_type == wire_type => reader.skip(wire_type)?,
            _ => reader.skip(wire_type)?,
        }
    }
    Ok(model)
}

fn decode_parameterized_variant(
    bytes: &[u8],
) -> Result<CursorParameterizedVariant, CursorWireError> {
    let mut reader = Reader::new(bytes);
    let mut variant = CursorParameterizedVariant::default();
    while !reader.is_empty() {
        let (field, wire_type) = reader.read_key()?;
        match (field, wire_type) {
            (1, 2) => {
                if variant.parameters.len() >= MAX_PARAMETERS_PER_VARIANT {
                    return Err(CursorWireError::TooManyItems);
                }
                variant
                    .parameters
                    .push(decode_model_parameter(reader.read_bytes()?)?);
            }
            (2, 2) => variant.display_name = Some(reader.read_string()?),
            (3, 0) => variant.is_max_mode = reader.read_varint()? != 0,
            (4, 0) => variant.is_default_max_config = Some(reader.read_varint()? != 0),
            (5, 0) => variant.is_default_non_max_config = Some(reader.read_varint()? != 0),
            (8, 2) => variant.display_name_outside_picker = Some(reader.read_string()?),
            (9, 2) => variant.variant_string_representation = Some(reader.read_string()?),
            _ => reader.skip(wire_type)?,
        }
    }
    Ok(variant)
}

fn decode_model_parameter(bytes: &[u8]) -> Result<CursorModelParameter, CursorWireError> {
    let mut reader = Reader::new(bytes);
    let mut parameter = CursorModelParameter::default();
    while !reader.is_empty() {
        let (field, wire_type) = reader.read_key()?;
        match (field, wire_type) {
            (1, 2) => parameter.id = reader.read_string()?,
            (2, 2) => parameter.value = reader.read_string()?,
            _ => reader.skip(wire_type)?,
        }
    }
    Ok(parameter)
}

fn encode_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn read_key(&mut self) -> Result<(u32, u8), CursorWireError> {
        let key = self.read_varint()?;
        let field = u32::try_from(key >> 3).map_err(|_| CursorWireError::Malformed)?;
        let wire_type = (key & 0x7) as u8;
        if field == 0 || matches!(wire_type, 3 | 4 | 6 | 7) {
            return Err(CursorWireError::Malformed);
        }
        Ok((field, wire_type))
    }

    fn read_varint(&mut self) -> Result<u64, CursorWireError> {
        let mut value = 0_u64;
        for index in 0..10 {
            let byte = *self
                .bytes
                .get(self.offset)
                .ok_or(CursorWireError::Malformed)?;
            self.offset += 1;
            if index == 9 && byte > 1 {
                return Err(CursorWireError::Malformed);
            }
            value |= u64::from(byte & 0x7f) << (index * 7);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(CursorWireError::Malformed)
    }

    fn read_bytes(&mut self) -> Result<&'a [u8], CursorWireError> {
        let length =
            usize::try_from(self.read_varint()?).map_err(|_| CursorWireError::Malformed)?;
        let end = self
            .offset
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(CursorWireError::Malformed)?;
        let bytes = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }

    fn read_string(&mut self) -> Result<String, CursorWireError> {
        String::from_utf8(self.read_bytes()?.to_vec()).map_err(|_| CursorWireError::InvalidText)
    }

    fn skip(&mut self, wire_type: u8) -> Result<(), CursorWireError> {
        match wire_type {
            0 => {
                self.read_varint()?;
            }
            1 => self.skip_bytes(8)?,
            2 => {
                self.read_bytes()?;
            }
            5 => self.skip_bytes(4)?,
            _ => return Err(CursorWireError::Malformed),
        }
        Ok(())
    }

    fn skip_bytes(&mut self, count: usize) -> Result<(), CursorWireError> {
        self.offset = self
            .offset
            .checked_add(count)
            .filter(|offset| *offset <= self.bytes.len())
            .ok_or(CursorWireError::Malformed)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field_varint(field: u64, value: u64, output: &mut Vec<u8>) {
        encode_varint((field << 3) | 0, output);
        encode_varint(value, output);
    }

    fn field_bytes(field: u64, value: &[u8], output: &mut Vec<u8>) {
        encode_varint((field << 3) | 2, output);
        encode_varint(value.len() as u64, output);
        output.extend_from_slice(value);
    }

    #[test]
    fn available_models_request_sets_parameter_and_markdown_flags() {
        assert_eq!(encode_available_models_request(), [0x28, 0x01, 0x38, 0x01]);
    }

    #[test]
    fn decodes_model_metadata_variants_and_unknown_fields() {
        let mut parameter = Vec::new();
        field_bytes(1, b"reasoning", &mut parameter);
        field_bytes(2, b"high", &mut parameter);

        let mut variant = Vec::new();
        field_bytes(1, &parameter, &mut variant);
        field_bytes(2, b"High", &mut variant);
        field_varint(3, 1, &mut variant);
        field_varint(4, 1, &mut variant);
        field_bytes(9, b"reasoning=high", &mut variant);

        let mut model = Vec::new();
        field_bytes(1, b"gpt-5.6", &mut model);
        field_varint(10, 1, &mut model);
        field_varint(14, 1, &mut model);
        field_varint(15, 272_000, &mut model);
        field_varint(16, 500_000, &mut model);
        field_bytes(17, b"GPT-5.6", &mut model);
        field_bytes(18, b"gpt-5.6", &mut model);
        field_varint(19, 1, &mut model);
        field_bytes(30, &variant, &mut model);
        field_bytes(31, b"ignored future field", &mut model);

        let mut response = Vec::new();
        field_bytes(2, &model, &mut response);
        field_bytes(100, b"unknown response field", &mut response);

        let decoded = decode_available_models_response(&response).expect("decode catalog");
        assert_eq!(decoded.len(), 1);
        let model = &decoded[0];
        assert_eq!(model.name, "gpt-5.6");
        assert_eq!(model.supports_images, Some(true));
        assert_eq!(model.context_token_limit, Some(272_000));
        assert_eq!(model.context_token_limit_for_max_mode, Some(500_000));
        assert_eq!(model.variants[0].parameters[0].id, "reasoning");
        assert_eq!(model.variants[0].parameters[0].value, "high");
        assert!(model.variants[0].is_max_mode);
    }

    #[test]
    fn rejects_invalid_wire_data_and_invalid_utf8() {
        assert_eq!(
            decode_available_models_response(&[0]),
            Err(CursorWireError::Malformed)
        );

        let mut model = Vec::new();
        field_bytes(1, &[0xff], &mut model);
        let mut response = Vec::new();
        field_bytes(2, &model, &mut response);
        assert_eq!(
            decode_available_models_response(&response),
            Err(CursorWireError::InvalidText)
        );
    }
}
