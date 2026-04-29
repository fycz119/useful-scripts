lm_eval \
  --model hf \
  --model_args '{"pretrained":"openai/gpt-oss-20b","dtype":"auto","chat_template_args":{"reasoning_effort":"low"},"enable_thinking": true,"think_end_token":200008}' \
  --device "cuda" \
  --tasks aime25 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs max_gen_toks=4048
