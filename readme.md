for cuda with faster-api must have:
cudnn_ops_infer64_8.dll
cudnn_cnn_infer64_8.dll
cublasLt64_12.dll

there's tradeoff between
compute_type
beam_size

compute_type="int8_float16", beam_size=5
compute_type="float16", beam_size=1

int16 isn't supported in current GPU
for ivrit-ai/whisper-large-v3-ct2: model.model.compute_type = float16

https://www.sefaria.org.il/api/v3/texts/Shevuot.2a?version=primary&version=translation&fill_in_missing_segments=1&return_format=wrap_all_entities

result.ori_dict["segments"][0]["words"]
result.ori_dict["segments"][0]["words"][0]['probability']
len(result.segments)

