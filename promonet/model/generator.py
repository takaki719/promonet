import ppgs
import torch

import promonet


###############################################################################
# ベースジェネレータの定義 (Base generator definition)
###############################################################################


class BaseGenerator(torch.nn.Module):
    """
    全てのジェネレータモデルの基底クラス。
    モデルアーキテクチャの選択、話者埋め込み、およびグローバル特徴量の準備に関する基本的な機能を実装します。
    """

    def __init__(self):
        """
        ジェネレータの初期化を行います。
        """
        super().__init__()

        # --- モデル選択 ---
        # 設定ファイル (promonet.MODEL) に基づいて、使用するボコーダモデルをインスタンス化します。
        if promonet.MODEL == 'fargan':
            self.model = promonet.model.FARGAN(
                promonet.NUM_FEATURES,       # 入力特徴量の次元数
                promonet.GLOBAL_CHANNELS)    # グローバル条件付けのチャンネル数
        elif promonet.MODEL == 'hifigan':
            self.model = promonet.model.HiFiGAN(
                promonet.NUM_FEATURES,
                promonet.GLOBAL_CHANNELS)
        elif promonet.MODEL == 'vocos':
            self.model = promonet.model.Vocos(
                promonet.NUM_FEATURES,
                promonet.GLOBAL_CHANNELS)
        else:
            # 定義されていないモデル名が指定された場合はエラーを送出
            raise ValueError(
                f'Generator model {promonet.MODEL} is not defined')

        # --- 話者埋め込み (Speaker embedding) ---
        # promonet.ZERO_SHOTがTrueの場合、事前学習済みのWavLMなどから抽出した話者特徴量ベクトルを
        # 線形層(Linear)でモデルの話者空間に射影します。これにより、学習時に未知だった話者にも対応できます（ゼロショット合成）。
        if promonet.ZERO_SHOT:
            self.speaker_embedding = torch.nn.Linear(
                promonet.WAVLM_EMBEDDING_CHANNELS, # 入力となるWavLM埋め込みの次元数
                promonet.SPEAKER_CHANNELS)         # モデル内部で扱う話者埋め込みの次元数
        # ZERO_SHOTがFalseの場合、学習データセット内の話者ID（整数）を埋め込みベクトルに変換する
        # Embedding層を使用します。これは学習済みの話者のみを再現する場合に用います。
        else:
            self.speaker_embedding = torch.nn.Embedding(
                promonet.NUM_SPEAKERS,      # 総話者数
                promonet.SPEAKER_CHANNELS)  # 話者埋め込みの次元数

        # --- 直前の音声サンプル用のバッファ ---
        # 自己回帰モデル(例: FARGAN)で、次のサンプルを生成する際に使用する直前の音声波形を格納するためのバッファです。
        # register_bufferにより、このテンソルはモデルのstate_dictに含まれますが、学習対象のパラメータとしては扱われません。
        self.register_buffer(
            'default_previous_samples',
            torch.zeros(1, 1, promonet.NUM_PREVIOUS_SAMPLES))

    def prepare_global_features(
        self,
        speakers,
        spectral_balance_ratios,
        loudness_ratios
    ):
        """
        グローバル条件付け特徴量（話者情報やデータ拡張に関する情報）を準備します。
        これらの特徴量は、音声全体のスタイルや品質を制御します。

        Args:
            speakers (torch.Tensor): 話者IDまたは話者埋め込みベクトル。
            spectral_balance_ratios (torch.Tensor): スペクトルバランスのデータ拡張比率。
            loudness_ratios (torch.Tensor): 音量のデータ拡張比率。

        Returns:
            torch.Tensor: チャンネル方向に結合されたグローバル特徴量テンソル。
        """
        # --- 話者情報のエンコード ---
        # 話者IDまたは埋め込みベクトルを、モデルで扱える形式の埋め込みベクトルに変換します。
        # unsqueeze(-1)で時間軸の次元を追加し、他の特徴量と結合できる形にします。
        global_features = self.speaker_embedding(speakers).unsqueeze(-1)

        # --- ピッチ拡張比率の追加 (オプション) ---
        # ピッチのデータ拡張が有効な場合、スペクトルバランス比率をグローバル特徴量に結合します。
        if promonet.AUGMENT_PITCH:
            global_features = torch.cat(
                (global_features, spectral_balance_ratios[:, None, None]),
                dim=1) # チャンネル次元(dim=1)で結合

        # --- 音量拡張比率の追加 (オプション) ---
        # 音量のデータ拡張が有効な場合、音量比率をグローバル特徴量に結合します。
        if promonet.AUGMENT_LOUDNESS:
            global_features = torch.cat(
                (global_features, loudness_ratios[:, None, None]),
                dim=1) # チャンネル次元(dim=1)で結合

        return global_features

    def remove_weight_norm(self):
        """
        モデルから重み正規化(weight normalization)を削除します。
        これは、TorchScriptへのエクスポートや推論の高速化のために行われます。
        """
        try:
            # HiFiGANなどのモデルにはremove_weight_normメソッドが実装されている
            self.model.remove_weight_norm()
        except AttributeError:
            # モデルにこのメソッドがない場合は何もしない
            pass


###############################################################################
# 提案ジェネレータの定義 (Proposed generator definition)
###############################################################################


class Generator(BaseGenerator):
    """
    PromoNetの主要なジェネレータ。BaseGeneratorを継承し、
    ピッチ、音量、周期性、PPGなどの詳細なローカル特徴量を処理する機能を追加します。
    また、推論用に特徴量をパック/アンパックする機能や、TorchScriptエクスポート機能も持ちます。
    """

    def __init__(self):
        """
        ジェネレータの初期化を行います。
        """
        super().__init__()

        # --- ピッチ埋め込み (オプション) ---
        # 'pitch'が入力特徴量に含まれ、かつPITCH_EMBEDDINGが有効な場合、
        # 離散化されたピッチビンIDを埋め込みベクトルに変換するための層を作成します。
        if 'pitch' in promonet.INPUT_FEATURES and promonet.PITCH_EMBEDDING:
            self.pitch_embedding = torch.nn.Embedding(
                promonet.PITCH_BINS,           # ピッチの離散化ビン数
                promonet.PITCH_EMBEDDING_SIZE) # ピッチ埋め込みの次元数

        # --- PPGスパース化のための閾値 ---
        # PPG (Phonetic PosteriorGrams) をスパース化（値の小さい要素を0にする）する手法が有効な場合、
        # その閾値をバッファとして登録します。
        if (
            promonet.SPARSE_PPG_METHOD is not None and
            ppgs.REPRESENTATION_KIND == 'ppg'
        ):
            ppg_threshold = torch.tensor(
                promonet.SPARSE_PPG_THRESHOLD,
                dtype=torch.float)
            self.register_buffer('ppg_threshold', ppg_threshold)

        # --- forwardメソッド内での条件分岐のためのフラグ ---
        # TorchScriptは、forwardメソッド内でのPythonのリストや辞書の動的な操作を嫌うため、
        # どの特徴量を使用するかをあらかじめブール値のフラグとして保持しておきます。
        self.use_pitch = 'pitch' in promonet.INPUT_FEATURES
        self.use_loudness = 'loudness' in promonet.INPUT_FEATURES
        self.use_periodicity = 'periodicity' in promonet.INPUT_FEATURES

        # --- 可変ピッチビンのための分布バッファ ---
        # TorchScriptはtry/exceptや動的な属性参照もサポートしないため、
        # 可変ピッチビンを使用する場合、ピッチの分布をあらかじめ読み込んでバッファに登録しておきます。
        if self.use_pitch and promonet.VARIABLE_PITCH_BINS:
            pitch_distribution = promonet.load.pitch_distribution()
            self.register_buffer('pitch_distribution', pitch_distribution)

    def forward(
        self,
        loudness,
        pitch,
        periodicity,
        ppg,
        speakers,
        spectral_balance_ratios,
        loudness_ratios,
        previous_samples
    ):
        """
        モデルの順伝播処理。入力特徴量から音声を合成します。

        Args:
            loudness (torch.Tensor): 音量特徴量。
            pitch (torch.Tensor): ピッチ(F0)特徴量。
            periodicity (torch.Tensor): 周期性特徴量。
            ppg (torch.Tensor): 音素事後確率 (Phonetic PosteriorGrams)。
            speakers (torch.Tensor): 話者IDまたは話者埋め込み。
            spectral_balance_ratios (torch.Tensor): スペクトルバランス拡張比率。
            loudness_ratios (torch.Tensor): 音量拡張比率。
            previous_samples (torch.Tensor): 自己回帰モデル用の直前の音声サンプル。

        Returns:
            torch.Tensor: 合成された音声波形。
        """
        # --- 入力特徴量の準備 ---
        # PPG, ピッチ, 音量, 周期性などのローカルな特徴量を結合し、モデルへの入力形式に整えます。
        features = self.prepare_features(loudness, pitch, periodicity, ppg)
        
        # --- グローバル特徴量の準備 ---
        # 話者情報などのグローバルな条件付け特徴量を準備します。
        global_features = self.prepare_global_features(
            speakers,
            spectral_balance_ratios,
            loudness_ratios)

        # --- 音声合成 ---
        # 準備した特徴量を使って、実際にモデルで音声波形を生成します。
        return self.model(features, global_features, previous_samples)

    def prepare_features(self, loudness, pitch, periodicity, ppg):
        """
        学習または推論のためのローカル入力特徴量を準備します。

        Args:
            loudness (torch.Tensor): 音量特徴量。
            pitch (torch.Tensor): ピッチ(F0)特徴量。
            periodicity (torch.Tensor): 周期性特徴量。
            ppg (torch.Tensor): 音素事後確率 (PPG)。

        Returns:
            torch.Tensor: モデルへの入力として整形された特徴量テンソル。
        """
        # --- PPGのスパース化 (オプション) ---
        # PPGの確率が低い音素を0にすることで、特徴量をスパース（疎）にします。
        # これにより、不要な音素の影響を減らす効果が期待されます。
        if (
            promonet.SPARSE_PPG_METHOD is not None and
            ppgs.REPRESENTATION_KIND == 'ppg'
        ):
            ppg = ppgs.sparsify(
                ppg,
                promonet.SPARSE_PPG_METHOD,
                self.ppg_threshold)

        features = ppg # まず、PPGをベースの特徴量とします。

        # --- ピッチ特徴量の追加 (オプション) ---
        if self.use_pitch:
            # ピッチ周波数(Hz)を、モデルが扱いやすい最小値(FMIN)と最大値(FMAX)の範囲にクリップします。
            hz = torch.clip(pitch, promonet.FMIN, promonet.FMAX)
            
            # PITCH_EMBEDDINGが有効な場合、ピッチを離散的なビンに変換し、埋め込みベクトルを取得します。
            if promonet.PITCH_EMBEDDING:
                # 可変ピッチビンを使用する場合 (よりデータに基づいた量子化)
                if promonet.VARIABLE_PITCH_BINS:
                    # 事前に計算したピッチ分布(pitch_distribution)を使い、各Hz値がどのビンに属するかを探索します。
                    bins = torch.searchsorted(self.pitch_distribution, hz)
                    bins = torch.clip(bins, 0, promonet.PITCH_BINS - 1)
                # 固定ピッチビンを使用する場合 (対数スケールでの均等量子化)
                else:
                    # F0を対数スケールに変換し、0-1の範囲に正規化します。
                    normalized = (
                        (torch.log2(hz) - promonet.LOG_FMIN) /
                        (promonet.LOG_FMAX - promonet.LOG_FMIN))
                    # 正規化された値をビンの数に合わせてスケーリングし、整数に変換します。
                    bins = (
                        (promonet.PITCH_BINS - 1) * normalized).to(torch.long)
                
                # ビンIDを埋め込みベクトルに変換し、次元の順番を(batch, channel, time)に合わせます。
                pitch_embeddings = self.pitch_embedding(bins).permute(0, 2, 1)
            # PITCH_EMBEDDINGが無効な場合、ピッチを連続値のまま正規化して使用します。
            else:
                pitch_embeddings = (
                    (torch.log2(hz)[:, None] - promonet.LOG_FMIN) /
                    (promonet.LOG_FMAX - promonet.LOG_FMIN))
            
            # 計算したピッチ特徴量を、既存の特徴量にチャンネル次元で結合します。
            features = torch.cat((features, pitch_embeddings), dim=1)

        # --- 音量特徴量の追加 (オプション) ---
        if self.use_loudness:
            bands = promonet.LOUDNESS_BANDS
            step = loudness.shape[-2] / bands
            # スペクトログラムを複数の周波数帯(バンド)に分割し、各バンドの平均音量を計算します。
            averaged = torch.stack(
                [
                    loudness[:, int(band * step):int((band + 1) * step)].mean(dim=-2)
                    for band in range(bands)
                ],
                dim=1)
            # 計算した平均音量を正規化します。
            normalized = promonet.preprocess.loudness.normalize(averaged)
            if normalized.ndim == 2:
                normalized = normalized[None] # バッチ次元を追加
            # 正規化された音量特徴量を結合します。
            features = torch.cat((features, normalized), dim=1)

        # --- 周期性特徴量の追加 (オプション) ---
        if self.use_periodicity:
            # 周期性(有声/無声の度合い)を特徴量として結合します。
            features = torch.cat((features, periodicity[:, None]), dim=1)

        # --- FARGAN用の周期特徴量の追加 ---
        # FARGANモデルは、ピッチの代わりに周期長(period)を入力として使用するため、
        # 周波数(pitch)から周期を計算して追加します。
        if promonet.MODEL == 'fargan':
            period = (
                promonet.SAMPLE_RATE /
                torch.clip(pitch, promonet.FMIN, promonet.FMAX))
            features = torch.cat((features, period[:, None]), dim=1)

        return features

    ###########################################################################
    # モデルのエクスポート関連 (Model exporting)
    ###########################################################################

    def export(self, output_file):
        """
        モデルをTorchScript形式でエクスポートします。

        Args:
            output_file (str): 保存先のファイルパス。
        """
        # 1. 重み正規化を削除して推論モードに最適化
        self.remove_weight_norm()

        # 2. パックされた推論メソッドを登録
        self.register()

        # 3. モデルをTorchScript化
        scripted = torch.jit.script(self)

        # 4. ファイルに保存
        scripted.save(output_file)

    @torch.jit.export
    def get_attributes(self):
        """
        (IRCAM nn~用) モデルの属性を取得します。現在は'none'を返します。
        """
        return ['none']

    @torch.jit.export
    def get_methods(self):
        """
        (IRCAM nn~用) エクスポートされた推論メソッドの名前のリストを取得します。
        """
        return self._methods

    def labels(self):
        """
        パックされた特徴量の各チャンネルのラベル(名前)をリストとして取得します。
        デバッグや外部ツールとの連携に役立ちます。

        Returns:
            list[str]: 各入力チャンネルのラベル。
        """
        labels = []

        # 音量
        labels += [
            f'loudness-{i}' for i in range(promonet.LOUDNESS_BANDS)]

        # ピッチ
        labels.append('pitch')

        # 周期性
        labels.append('periodicity')

        # PPG
        labels += [
            f'ppg-{i} ({ppgs.PHONEMES[i]})' # 各PPGチャンネルに音素名を付与
            for i in range(promonet.PPG_CHANNELS)]

        # 話者
        labels.append('speaker')

        # スペクトルバランス (データ拡張用)
        labels.append('spectral balance')

        # 音量比率 (データ拡張用)
        labels.append('loudness ratio')

        return labels

    def pack_features(
        self,
        loudness,
        pitch,
        periodicity,
        ppg,
        speakers,
        spectral_balance_ratios,
        loudness_ratios
    ):
        """
        個別の特徴量テンソルを、フレーム解像度の単一のテンソルに「パック」します。
        これは `unpack_features` の逆の操作です。

        Returns:
            torch.Tensor: 全ての特徴量がチャンネル次元で結合された単一のテンソル。
        """
        # (batch, channel, time) の形状を持つ空のテンソルから開始
        features = torch.zeros((loudness.shape[0], 0, loudness.shape[2]))

        # 音量
        if self.use_loudness:
            averaged = promonet.preprocess.loudness.band_average(loudness)
            features = torch.cat((features, averaged), dim=1)

        # ピッチ
        if self.use_pitch:
            features = torch.cat((features, pitch), dim=1)

        # 周期性
        if self.use_periodicity:
            features = torch.cat((features, periodicity), dim=1)

        # PPG
        if (
            promonet.SPARSE_PPG_METHOD is not None and
            ppgs.REPRESENTATION_KIND == 'ppg'
        ):
            ppg = ppgs.sparsify(
                ppg,
                promonet.SPARSE_PPG_METHOD,
                self.ppg_threshold)
        features = torch.cat((features, ppg), dim=1)

        # 話者 (時間方向にリピートして次元を合わせる)
        speakers = speakers[:, None, None].repeat(1, 1, features.shape[-1])
        features = torch.cat((features, speakers.to(torch.float)), dim=1)

        # スペクトルバランス (時間方向にリピート)
        if promonet.AUGMENT_PITCH:
            spectral_balance_ratios = \
                spectral_balance_ratios[:, None, None].repeat(
                    1, 1, features.shape[-1])
            features = torch.cat((features, spectral_balance_ratios), dim=1)

        # 音量比率 (時間方向にリピート)
        if promonet.AUGMENT_LOUDNESS:
            loudness_ratios = loudness_ratios[:, None, None].repeat(
                1, 1, features.shape[-1])
            features = torch.cat((features, loudness_ratios), dim=1)

        return features

    @torch.jit.export
    def packed_inference(self, x):
        """
        TorchScriptとしてエクスポートされる推論用のメイン関数。
        単一の「パック済み」特徴量テンソルを受け取り、音声を合成します。

        Args:
            x (torch.Tensor): フレーム解像度のパックされた入力特徴量テンソル。
                               形状は (batch, num_features, time_frames)。

        Returns:
            torch.Tensor: 合成された音声波形。形状は (batch, 1, samples)。
        """
        # 1. パックされたテンソルxを、個別の特徴量にアンパック（展開）します。
        (
            loudness,
            pitch,
            periodicity,
            ppg,
            speakers,
            spectral_balance_ratios,
            loudness_ratios
        ) = self.unpack_features(x)

        # 2. アンパックした特徴量を使って、メインのforwardメソッドを呼び出します。
        # torch.jitはキーワード引数のアンパックをサポートしないため、引数を直接渡します。
        # `default_previous_samples`は、自己回帰を行わない推論ではゼロ埋めされたものが使われます。
        return self(
            loudness,
            pitch,
            periodicity,
            ppg,
            speakers,
            spectral_balance_ratios,
            loudness_ratios,
            self.default_previous_samples
        ).to(torch.float) # 出力形式をfloatに変換

    def register(
        self,
        method_name: str = 'packed_inference',
        test_buffer_size: int = 8192
    ):
        """
        クラスメソッドをIRCAMのnn~などの外部ツールで使用できるように登録します。
        メタデータ（入力チャンネル数、ホップサイズなど）をバッファに保存し、
        チャンネルラベルを設定します。
        """
        # 1. 各入力チャンネルのセマンティックなラベルを取得します。
        labels = self.labels()

        # 2. 入出力サイズなどのメタデータを格納するバッファを作成します。
        # [入力チャンネル数, ホップサイズ, 出力チャンネル数, 不明]
        self.register_buffer(
            f'{method_name}_params',
            torch.tensor([len(labels), promonet.HOPSIZE, 1, 1]))

        # 3. 各入出力チャンネルにラベルを設定します。
        setattr(self, f'{method_name}_input_labels', labels)
        setattr(self, f'{method_name}_output_labels', ['output audio'])

        # 4. パックされた推論が正しく動作するかをテストします。
        # ダミーの入力データを作成
        x = torch.zeros(1, len(labels), test_buffer_size // promonet.HOPSIZE)
        # 推論メソッドを実行
        y = getattr(self, method_name)(x)
        # 出力の形状と型が期待通りであることを確認
        assert (
            tuple(y.shape) == (1, 1, test_buffer_size) and
            y.dtype == torch.float)

        # 5. 登録するメソッドの名前をリストとして保持します。
        self._methods = [method_name]

    def unpack_features(self, x):
        """
        単一のパックされたフレーム解像度テンソルを、個別の特徴量にアンパックします。
        この順序は`labels()`や`pack_features()`と一致している必要があります。

        Args:
            x (torch.Tensor): パックされた特徴量テンソル。

        Returns:
            tuple: アンパックされた個別の特徴量テンソルのタプル。
        """
        i = 0 # 現在のチャンネル位置を示すインデックス

        # 音量
        loudness = x[:, i:i + promonet.LOUDNESS_BANDS]
        i += promonet.LOUDNESS_BANDS

        # ピッチ
        pitch = x[:, i:i + 1].squeeze(1) # チャンネル次元を削除
        i += 1

        # 周期性
        periodicity = x[:, i:i + 1].squeeze(1) # チャンネル次元を削除
        i += 1

        # PPG
        ppg = x[:, i:i + promonet.PPG_CHANNELS]
        i += promonet.PPG_CHANNELS

        # 話者 (話者IDはlong型に変換)
        speakers = x[:, i:i + 1, 0].to(torch.long).squeeze(1)
        i += 1

        # スペクトルバランス
        spectral_balance_ratios = x[:, i:i + 1, 0].squeeze(1)
        i += 1

        # 音量比率
        loudness_ratios = x[:, i:i + 1, 0].squeeze(1)
        i += 1

        return (
            loudness,
            pitch,
            periodicity,
            ppg,
            speakers,
            spectral_balance_ratios,
            loudness_ratios)


###############################################################################
# メルスペクトログラム用ボコーダ (Mel vocoder)
###############################################################################


class MelGenerator(BaseGenerator):
    """
    メルスペクトログラムから直接音声を生成する、よりシンプルなジェネレータ。
    伝統的なText-to-Speechパイプラインのボコーダ部分に相当します。
    """

    def forward(
        self,
        spectrograms,
        speakers,
        spectral_balance_ratios,
        loudness_ratios,
        previous_samples
    ):
        """
        順伝播処理。メルスペクトログラムから音声を合成します。

        Args:
            spectrograms (torch.Tensor): 入力となる線形またはメルスペクトログラム。
            (他の引数はBaseGeneratorと同様)

        Returns:
            torch.Tensor: 合成された音声波形。
        """
        # 1. 入力スペクトログラムをモデルが要求する特徴量形式に変換します。
        features = self.prepare_features(spectrograms)
        
        # 2. グローバル特徴量を準備します。
        global_features = self.prepare_global_features(
            speakers,
            spectral_balance_ratios,
            loudness_ratios)

        # 3. 音声合成を実行します。
        # CARGANモデルはprevious_samplesを必要とする場合があります。
        if promonet.MODEL == 'cargan':
            return self.model(features, global_features, previous_samples)
        # 他の多くのメルボコーダは自己回帰的ではありません。
        return self.model(features, global_features)

    def prepare_features(self, spectrograms):
        """
        学習または推論のための入力特徴量（メルスペクトログラム）を準備します。
        """
        # 1. 線形スペクトログラムをメルスケールに変換します。
        features = promonet.preprocess.spectrogram.linear_to_mel(
            spectrograms)

        # 2. メルスペクトログラムをスパース化する (オプション)
        # ダイナミックレンジ圧縮の対数閾値を加算することで、非常に小さい値をクリッピングします。
        if promonet.SPARSE_MELS:
            features += promonet.LOG_DYNAMIC_RANGE_COMPRESSION_THRESHOLD

        return features
