⏺ それぞれのコマンドの詳細機能を説明します：

  🎵 1. 前処理 (preprocess)

  python -m promonet.preprocess --files japanese_audio.wav --gpu 0

  何をする？: 日本語音声から編集用の特徴量を抽出

  抽出される特徴量:
  - 音量 (loudness): 8帯域のA重み付け音量
  - ピッチ (pitch): 基本周波数（声の高さ）
  - 周期性 (periodicity): 有声/無声の判定
  - PPG: 日本語音素の確率分布
  - テキスト: Whisperによる日本語音声認識結果

  出力: japanese_audio-loudness.pt, japanese_audio-pitch.pt など

  ✂️ 2. 編集 (edit)

  python -m promonet.edit --pitch_shift_cents 100 --time_stretch_ratio 1.2 \
      --loudness_files japanese_audio-loudness.pt \
      --pitch_files japanese_audio-pitch.pt \
      --periodicity_files japanese_audio-periodicity.pt \
      --ppg_files japanese_audio-ppg.pt \
      --output_prefixes edited_japanese

  何をする？: 音声の韻律を操作

  編集可能な内容:
  - ピッチシフト (--pitch_shift_cents 100): 声を100セント（半音）高くする
  - テンポ変更 (--time_stretch_ratio 1.2): 1.2倍速にする（20%高速化）
  - 音量調整 (--loudness_scale_db 6): 音量を6dB上げる
  - 発音編集: PPG操作による日本語発音の微調整

  出力: 編集された特徴量ファイル群

  🎤 3. 合成 (synthesize)

  python -m promonet.synthesize \
      --loudness_files edited_japanese-loudness.pt \
      --pitch_files edited_japanese-pitch.pt \
      --periodicity_files edited_japanese-periodicity.pt \
      --ppg_files edited_japanese-ppg.pt \
      --output_files edited_japanese.wav --gpu 0

  何をする？: 編集された特徴量から音声を生成

  合成機能:
  - 高品質音声生成: HiFi-GANなどのニューラル音声合成
  - 話者性保持: 元の話者の声質を維持
  - 自然性: 編集後も自然な日本語音声

  🔄 実用例

  # 例1: 日本語音声を高くして早口にする
  python -m promonet.preprocess --files original.wav --gpu 0
  python -m promonet.edit \
      --loudness_files original-loudness.pt \
      --pitch_files original-pitch.pt \
      --periodicity_files original-periodicity.pt \
      --ppg_files original-ppg.pt \
      --output_prefixes high_fast \
      --pitch_shift_cents 200 \
      --time_stretch_ratio 1.5
  python -m promonet.synthesize \
      --loudness_files high_fast-loudness.pt \
      --pitch_files high_fast-pitch.pt \
      --periodicity_files high_fast-periodicity.pt \
      --ppg_files high_fast-ppg.pt \
      --output_files high_fast_voice.wav --gpu 0