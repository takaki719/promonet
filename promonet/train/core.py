import math

import GPUtil
import matplotlib.pyplot as plt
import torch
import torchutil

import promonet


###############################################################################
# 学習 (Training)
###############################################################################


@torchutil.notify('train')
def train(
    directory,
    dataset=promonet.TRAINING_DATASET,
    train_partition='train',
    valid_partition='valid',
    adapt_from=None,
    gpu=None
):
    """
    モデルの学習を実行します。

    Args:
        directory (str): チェックポイントやログを保存するディレクトリ。
        dataset (str): 使用するデータセット名。
        train_partition (str): 学習に使用するデータセットのパーティション名。
        valid_partition (str): 検証に使用するデータセットのパーティション名。
        adapt_from (str, optional): 適応学習（ファインチューニング）の元となる学習済みモデルのディレクトリ。
        gpu (int, optional): 使用するGPUのID。Noneの場合はCPUを使用します。
    """
    # Matplotlibが評価中に多数の図を同時に開くことに関する警告を抑制します。
    # 図は適切に閉じられるため、この警告は不要です。
    plt.rcParams.update({'figure.max_open_warning': 100})

    # PyTorchが使用するデバイス（CPUまたはGPU）を取得します。
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #######################
    # データローダの作成 #
    #######################

    torch.manual_seed(promonet.RANDOM_SEED) # 再現性のために乱数シードを固定
    # 学習用データローダ
    train_loader = promonet.data.loader(
        dataset,
        train_partition,
        adapt_from is not None, # 適応学習モードか否か
        gpu)
    # 検証用データローダ
    valid_loader = promonet.data.loader(
        dataset,
        valid_partition,
        adapt_from is not None,
        gpu)

    #################
    # モデルの作成 #
    #################

    # SPECTROGRAM_ONLYがTrueの場合、メルスペクトログラムから音声を生成するシンプルなモデルを使用
    if promonet.SPECTROGRAM_ONLY:
        generator = promonet.model.MelGenerator().to(device)
    # そうでなければ、PPGやピッチなど詳細な特徴量から生成するメインのモデルを使用
    else:
        generator = promonet.model.Generator().to(device)
    # 識別器モデルを作成
    discriminators = promonet.model.Discriminator().to(device)

    #####################
    # オプティマイザの作成 #
    #####################

    # 識別器用のオプティマイザ
    discriminator_optimizer = promonet.OPTIMIZER(discriminators.parameters())
    # 生成器用のオプティマイザ
    generator_optimizer = promonet.OPTIMIZER(generator.parameters())

    ##############################
    # チェックポイントからの読み込み #
    ##############################

    # 読み込むべき最新のチェックポイントパスを取得
    # adapt_fromが指定されていればそちらから、なければ現在の学習ディレクトリから探す
    generator_path = torchutil.checkpoint.latest_path(
        directory if adapt_from is None else adapt_from,
        'generator-*.pt')
    discriminator_path = torchutil.checkpoint.latest_path(
        directory if adapt_from is None else adapt_from,
        'discriminator-*.pt')

    # 両方のチェックポイントが存在すれば、学習を再開する
    if generator_path and discriminator_path:
        # 生成器とそのオプティマイザの状態を読み込む
        (
            generator,
            generator_optimizer,
            state
        ) = torchutil.checkpoint.load(
            generator_path,
            generator,
            generator_optimizer
        )
        step, epoch = state['step'], state['epoch'] # 現在のステップとエポック数を復元

        # 識別器とそのオプティマイザの状態を読み込む
        (
            discriminators,
            discriminator_optimizer,
            _
        ) = torchutil.checkpoint.load(
            discriminator_path,
            discriminators,
            discriminator_optimizer
        )
    # チェックポイントがない場合は、最初から学習を開始する
    else:
        step, epoch = 0, 0

    #########
    # 学習 #
    #########

    # 総学習ステップ数を設定
    if adapt_from: # 適応学習の場合
        steps = promonet.STEPS + promonet.ADAPTATION_STEPS
    else: # 通常の学習の場合
        steps = promonet.STEPS

    # 自動混合精度（AMP）のための勾配スケーラを作成
    scaler = torch.cuda.amp.GradScaler()

    # スペクトル収束損失をセットアップ（オプション）
    if promonet.SPECTRAL_CONVERGENCE_LOSS:
        spectral_convergence = \
            promonet.loss.MultiResolutionSpectralConvergence(device)

    # 学習の進捗バーをセットアップ
    progress = torchutil.iterator(
        range(step, steps),
        f'{"Train" if adapt_from is None else "Adapt"}ing {promonet.CONFIG}',
        initial=step,
        total=steps)
    
    # メインの学習ループ
    while step < steps:
        # データローダのサンプラーに現在のエポック数を設定（分散学習で重要）
        train_loader.batch_sampler.set_epoch(epoch)

        for batch in train_loader:
            # --- バッチデータの準備 ---
            # バッチを各特徴量にアンパック
            (
                _,
                loudness,
                pitch,
                periodicity,
                ppg,
                speakers,
                spectral_balance_ratios,
                loudness_ratios,
                spectrograms,
                audio,
                _
            ) = batch

            # 音声が短すぎるサンプルはスキップ
            if audio.shape[-1] < promonet.CHUNK_SIZE:
                continue

            # 全てのテンソルを学習デバイス（GPUなど）にコピー
            (
                loudness,
                pitch,
                periodicity,
                ppg,
                speakers,
                spectral_balance_ratios,
                loudness_ratios,
                spectrograms,
                audio
            ) = (
                item.to(device) for item in
                (
                    loudness,
                    pitch,
                    periodicity,
                    ppg,
                    speakers,
                    spectral_balance_ratios,
                    loudness_ratios,
                    spectrograms,
                    audio
                )
            )

            # --- 生成器への入力データを準備 ---
            # 自己回帰モデルのために、先行する音声サンプルを準備
            if promonet.MODEL == 'cargan':
                previous_samples = audio[..., :promonet.CARGAN_INPUT_SIZE]
                slice_frames = promonet.CARGAN_INPUT_SIZE // promonet.HOPSIZE
            elif promonet.MODEL == 'fargan':
                previous_samples = audio[
                    ...,
                    :promonet.HOPSIZE * promonet.FARGAN_PREVIOUS_FRAMES]
                slice_frames = 0
            else: # 自己回帰でないモデルの場合
                previous_samples = torch.zeros(
                    1, 1, promonet.NUM_PREVIOUS_SAMPLES,
                    dtype=audio.dtype,
                    device=audio.device)
                slice_frames = 0
            
            # モデルの種類に応じて入力をまとめる
            if promonet.SPECTROGRAM_ONLY:
                generator_input = (
                    spectrograms[..., slice_frames:],
                    speakers,
                    spectral_balance_ratios,
                    loudness_ratios,
                    previous_samples)
            else: # メインモデルの場合
                generator_input = (
                    loudness[..., slice_frames:],
                    pitch[..., slice_frames:],
                    periodicity[..., slice_frames:],
                    ppg[..., slice_frames:],
                    speakers,
                    spectral_balance_ratios,
                    loudness_ratios,
                    previous_samples)

            #######################
            # 識別器の学習 #
            #######################

            # 混合精度で計算を行うコンテキスト
            with torch.autocast(device.type):
                # 生成器で偽の音声（generated）を生成
                generated = generator(*generator_input)

                # 自己回帰モデルの場合、先行サンプルと結合して評価
                if promonet.MODEL == 'cargan':
                    generated = torch.cat((previous_samples, generated), dim=1)
                elif promonet.MODEL == 'fargan':
                    generated = torch.cat(
                        (
                            previous_samples,
                            generated[..., previous_samples.shape[-1]:]
                        ),
                        dim=2)

                # 識別器の学習は指定されたステップから開始
                if step >= promonet.DISCRIMINATOR_START_STEP:
                    # 識別器に本物の音声(audio)と偽の音声(generated)を入力
                    real_logits, fake_logits, _, _ = discriminators(
                        audio,
                        generated.detach()) # 生成器の勾配計算を停止

                    # 識別器の損失を計算
                    (
                        discriminator_losses,
                        real_discriminator_losses,
                        fake_discriminator_losses
                    ) = promonet.loss.discriminator(
                        [logit.float() for logit in real_logits],
                        [logit.float() for logit in fake_logits])

            # 識別器の逆伝播とパラメータ更新
            if step >= promonet.DISCRIMINATOR_START_STEP:
                discriminator_optimizer.zero_grad()
                scaler.scale(discriminator_losses).backward() # スケーリングされた損失で逆伝播
                scaler.step(discriminator_optimizer) # オプティマイザで更新

            ###################
            # 生成器の学習 #
            ###################

            with torch.autocast(device.type):
                # 敵対的損失の計算は指定されたステップから開始
                if step >= promonet.ADVERSARIAL_LOSS_START_STEP:
                    # 識別器に再度入力（今度は生成器の勾配を計算するためdetachしない）
                    (
                        _,
                        fake_logits,
                        real_feature_maps,
                        fake_feature_maps
                    ) = discriminators(audio, generated)

                # --- 各種の生成器損失を計算 ---
                generator_losses = 0.

                # メルスペクトログラム損失（オプション）
                if promonet.MEL_LOSS:
                    
                    # ターゲットとなるメルスペクトログラムを計算            
                    log_dynamic_range_compression_threshold = (
                        promonet.LOG_DYNAMIC_RANGE_COMPRESSION_THRESHOLD
                        if promonet.SPARSE_MEL_LOSS else None)

                    # 生成された音声からメルスペクトログラムを計算
                    mels = promonet.preprocess.spectrogram.linear_to_mel(
                        spectrograms,
                        log_dynamic_range_compression_threshold)

                    # L1損失を計算し、重みをかけて加算
                    generated_mels = promonet.preprocess.spectrogram.from_audio(
                        generated,
                        True,
                        log_dynamic_range_compression_threshold)

                    # Maybe shift so clipping bound is zero
                    if promonet.SPARSE_MEL_LOSS:
                        mels += promonet.LOG_DYNAMIC_RANGE_COMPRESSION_THRESHOLD
                        generated_mels += \
                            promonet.LOG_DYNAMIC_RANGE_COMPRESSION_THRESHOLD

                    # Mel loss
                    mel_loss = torch.nn.functional.l1_loss(
                        mels,
                        generated_mels)
                    generator_losses += promonet.MEL_LOSS_WEIGHT * mel_loss

                # スペクトル収束損失（オプション）
                if promonet.SPECTRAL_CONVERGENCE_LOSS:
                    spectral_loss = spectral_convergence(generated, audio)
                    generator_losses += spectral_loss

                # 生波形損失（オプション）
                if promonet.SIGNAL_LOSS:
                    signal_loss = promonet.loss.signal(audio, generated)
                    generator_losses += promonet.SIGNAL_LOSS_WEIGHT * signal_loss

                if step >= promonet.ADVERSARIAL_LOSS_START_STEP:
                    # 特徴量マッチング損失（識別器の中間層出力のL1損失）
                    feature_matching_loss = promonet.loss.feature_matching(
                        real_feature_maps,
                        fake_feature_maps)
                    generator_losses += (
                        promonet.FEATURE_MATCHING_LOSS_WEIGHT *
                        feature_matching_loss)

                    # 敵対的損失（生成器が識別器を騙せたかの指標）
                    adversarial_loss, adversarial_losses = \
                        promonet.loss.generator(
                            [logit.float() for logit in fake_logits])
                    generator_losses += \
                        promonet.ADVERSARIAL_LOSS_WEIGHT * adversarial_loss

            # --- 生成器のパラメータ更新 ---
            generator_optimizer.zero_grad() # 勾配をゼロに初期化
            
            scaler.scale(generator_losses).backward()

            # 勾配の統計情報を監視
            gradient_statistics = torchutil.gradients.stats(generator)
            torchutil.tensorboard.update(
                directory,
                step,
                scalars=gradient_statistics)

            # 勾配クリッピング（オプション）
            if promonet.GRADIENT_CLIP_GENERATOR is not None:
                # 勾配の最大値が閾値を超えた場合にクリッピングを実行
                max_grad = max(
                    gradient_statistics['gradients/max'],
                    math.abs(gradient_statistics['gradients/min']))
                if max_grad > promonet.GRADIENT_CLIP_GENERATOR:

                    # スケーリングを元に戻す
                    scaler.unscale_(generator_optimizer)
                    # 勾配をクリップ
                    torch.nn.utils.clip_grad_norm_(
                        generator.parameters(),
                        promonet.GRADIENT_CLIP_GENERATOR,
                        norm_type="inf")

            # パラメータの更新
            scaler.step(generator_optimizer)
            # 勾配スケーラの更新
            scaler.update()

            ###########
            # ログ記録 #
            ###########

            # 一定間隔で評価とログ記録を実行
            if step % promonet.EVALUATION_INTERVAL == 0:
                # GPUメモリ使用量をログに記録
                torchutil.tensorboard.update(
                    directory,
                    step,
                    scalars=torchutil.cuda.utilization(device, 'MB'))

                # 各種学習損失をTensorBoardに記録
                scalars = {'loss/generator/total': generator_losses}
                if promonet.MEL_LOSS: scalars.update({'loss/generator/mels': mel_loss})
                if promonet.SIGNAL_LOSS: scalars.update({'loss/generator/signal': signal_loss})
                if promonet.SPECTRAL_CONVERGENCE_LOSS: scalars.update({'loss/generator/spectral-convergence': spectral_loss})
                if step >= promonet.ADVERSARIAL_LOSS_START_STEP:
                    scalars.update({
                    'loss/discriminator/total': discriminator_losses,
                    'loss/generator/feature-matching':
                        feature_matching_loss})
                    scalars.update(
                        {f'loss/generator/adversarial-{i:02d}': value
                        for i, value in enumerate(adversarial_losses)})
                    scalars.update(
                        {f'loss/discriminator/real-{i:02d}': value
                        for i, value in enumerate(real_discriminator_losses)})
                    scalars.update(
                        {f'loss/discriminator/fake-{i:02d}': value
                        for i, value in enumerate(fake_discriminator_losses)})
                    # ... (その他の詳細な損失も記録)
                torchutil.tensorboard.update(directory, step, scalars=scalars)

                # 検証データで評価を実行
                with torchutil.inference.context(generator):
                    evaluation_steps = (
                        None if step == steps
                        else promonet.DEFAULT_EVALUATION_STEPS)
                    evaluate(
                        directory,
                        step,
                        generator,
                        valid_loader,
                        gpu,
                        evaluation_steps)

            ###################
            # チェックポイントの保存 #
            ###################

            if step and step % promonet.CHECKPOINT_INTERVAL == 0:
                torchutil.checkpoint.save(
                    directory / f'generator-{step:08d}.pt',
                    generator,
                    generator_optimizer,
                    step=step,
                    epoch=epoch)
                torchutil.checkpoint.save(
                    directory / f'discriminator-{step:08d}.pt',
                    discriminators,
                    discriminator_optimizer,
                    step=step,
                    epoch=epoch)

            ################
            # 学習終了判定 #
            ################

            if step >= steps:
                break
            
            # GPU温度が80度を超えたら緊急停止
            if any(gpu.temperature > 80. for gpu in GPUtil.getGPUs()):
                raise RuntimeError(
                    f'GPU is overheating. Terminating training.')

            # ステップ数をインクリメント
            step += 1

            # 進捗バーを更新
            progress.update() 
        epoch += 1 # エポック数をインクリメント

    # 進捗バーを閉じる
    progress.close()

    # 最終的なモデルを保存
    torchutil.checkpoint.save(
        directory / f'generator-{step:08d}.pt',
        generator,
        generator_optimizer,
        step=step,
        epoch=epoch)
    torchutil.checkpoint.save(
        directory / f'discriminator-{step:08d}.pt',
        discriminators,
        discriminator_optimizer,
        step=step,
        epoch=epoch)


###############################################################################
# 評価 (Evaluation)
###############################################################################


def evaluate(directory, step, generator, loader, gpu, evaluation_steps=None):
    """
    モデルの性能評価を実行します。
    """
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # --- 評価指標のセットアップ ---
    # 各評価条件（再構成、ピッチシフトなど）ごとにメトリクスを計算するクラスを準備
    metrics = {'reconstruction': promonet.evaluate.Metrics()}
    ratios = [
        f'{int(ratio * 100):03d}' for ratio in promonet.EVALUATION_RATIOS]
    if 'pitch' in promonet.INPUT_FEATURES:
        metrics.update({
            f'shifted-{ratio}': promonet.evaluate.Metrics()
            for ratio in ratios})
    if 'ppg' in promonet.INPUT_FEATURES:
        metrics.update({
            f'stretched-{ratio}': promonet.evaluate.Metrics()
            for ratio in ratios})
    if 'loudness' in promonet.INPUT_FEATURES:
        metrics.update({
            f'scaled-{ratio}': promonet.evaluate.Metrics()
            for ratio in ratios})

    # TensorBoardに保存するための音声、図、スカラー値を初期化
    waveforms, figures, scalars = {}, {}, {}

    # データ拡張用の比率は評価時はデフォルト値(1.0)を使用
    spectral_balance_ratios = torch.ones(1, dtype=torch.float, device=device)
    loudness_ratios = torch.ones(1, dtype=torch.float, device=device)

    for i, batch in enumerate(loader):
        # --- バッチデータの準備 ---
        (
            _,
            loudness,
            pitch,
            periodicity,
            ppg,
            speakers,
            _,
            _,
            spectrogram,
            audio,
            _
        ) = batch
        # 必要なデータをデバイスにコピー
        (
            loudness,
            pitch,
            periodicity,
            ppg,
            speakers,
            spectrogram,
            audio
        ) = (
            item.to(device) for item in (
                loudness,
                pitch,
                periodicity,
                ppg,
                speakers,
                spectrogram,
                audio
            )
        )
        
        # グローバル特徴量をまとめる
        global_features = (
            speakers,
            spectral_balance_ratios,
            loudness_ratios,
            generator.default_previous_samples)

        # ターゲット音声と生成音声の長さを合わせる
        trim = audio.shape[-1] % promonet.HOPSIZE
        if trim > 0:
            audio = audio[..., :-trim]
        
        # 最初の評価時に元の音声を保存
        if step == 0:
            waveforms[f'original/{i:02d}-audio'] = audio[0]

        ##################
        # 1. 再構成評価 #
        ##################
        # 元の特徴量から音声を生成し、元の音声とどれだけ近いかを評価

        # 生成器への入力
        if promonet.SPECTROGRAM_ONLY:
            generator_input = (spectrogram, *global_features)
        else:
            generator_input = (
                loudness,
                pitch,
                periodicity,
                ppg,
                *global_features)
        generated = generator(*generator_input)
        key = f'reconstruction/{i:02d}'
        waveforms[f'{key}-audio'] = generated[0]
        
        # 生成音声から特徴量を再抽出
        (
            predicted_loudness,
            predicted_pitch,
            predicted_periodicity,
            predicted_ppg
        ) = promonet.preprocess.from_audio(generated[0], gpu=gpu)
        
        # 特徴量の比較プロットを生成
        if i < promonet.PLOT_EXAMPLES:
            figures[key] = promonet.plot.from_features(
                generated,
                promonet.preprocess.loudness.band_average(predicted_loudness, 1),
                predicted_pitch,
                predicted_periodicity,
                predicted_ppg,
                promonet.preprocess.loudness.band_average(loudness, 1),
                pitch,
                periodicity,
                ppg)
        
        # メトリクスを更新
        metrics[key.split('/')[0]].update(
            loudness,
            pitch,
            periodicity,
            ppg,
            predicted_loudness,
            predicted_pitch,
            predicted_periodicity,
            predicted_ppg)

        ######################
        # 2. ピッチシフト評価 #
        ######################
        if 'pitch' in promonet.INPUT_FEATURES:
            for ratio in promonet.EVALUATION_RATIOS:

                # Shift pitch
                shifted_pitch = ratio * pitch

                # Generate pitch-shifted speech
                generator_input = (
                    loudness,
                    shifted_pitch,
                    periodicity,
                    ppg,
                    *global_features)
                shifted = generator(*generator_input)

                # Get prosody features
                (
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg
                ) = promonet.preprocess.from_audio(shifted[0], gpu=gpu)

                # Log pitch-shifted audio
                key = f'shifted-{int(100 * ratio):03d}/{i:02d}'
                waveforms[f'{key}-audio'] = shifted[0]

                # Log prosody figure
                if i < promonet.PLOT_EXAMPLES:
                    figures[key] = promonet.plot.from_features(
                        shifted,
                        promonet.preprocess.loudness.band_average(predicted_loudness, 1),
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_ppg,
                        promonet.preprocess.loudness.band_average(loudness, 1),
                        shifted_pitch,
                        periodicity,
                        ppg)

                # Update metrics
                metrics[key.split('/')[0]].update(
                    loudness,
                    shifted_pitch,
                    periodicity,
                    ppg,
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg)

        #######################
        # 3. タイムストレッチ評価 #
        #######################
        if 'ppg' in promonet.INPUT_FEATURES:
            for ratio in promonet.EVALUATION_RATIOS:
                # 特徴量をタイムストレッチ
                (
                    stretched_loudness,
                    stretched_pitch,
                    stretched_periodicity,
                    stretched_ppg
                ) = promonet.edit.from_features(
                    loudness,
                    pitch,
                    periodicity,
                    ppg,
                    time_stretch_ratio=ratio)

                # Generate
                generator_input = (
                    stretched_loudness,
                    stretched_pitch,
                    stretched_periodicity,
                    stretched_ppg,
                    *global_features)
                stretched = generator(*generator_input)

                # Get prosody features
                (
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg
                ) = promonet.preprocess.from_audio(stretched[0], gpu=gpu)

                # Log time-stretched audio
                key = f'stretched-{int(ratio * 100):03d}/{i:02d}'
                waveforms[f'{key}-audio'] = stretched[0]

                # Log prosody figure
                if i < promonet.PLOT_EXAMPLES:
                    figures[key] = promonet.plot.from_features(
                        stretched,
                        promonet.preprocess.loudness.band_average(predicted_loudness, 1),
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_ppg,
                        promonet.preprocess.loudness.band_average(stretched_loudness, 1),
                        stretched_pitch,
                        stretched_periodicity,
                        stretched_ppg)

                # Update metrics
                metrics[key.split('/')[0]].update(
                    stretched_loudness,
                    stretched_pitch,
                    stretched_periodicity,
                    stretched_ppg,
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg)

        ####################
        # 4. 音量スケール評価 #
        ####################
        if 'loudness' in promonet.INPUT_FEATURES:
            for ratio in promonet.EVALUATION_RATIOS:
                # 音量をスケール
                scaled_loudness = \
                    loudness + promonet.convert.ratio_to_db(ratio)

                # Generate loudness-scaled speech
                generator_input = (
                    scaled_loudness,
                    pitch,
                    periodicity,
                    ppg,
                    *global_features)
                scaled = generator(*generator_input)

                # Get prosody features
                (
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg
                ) = promonet.preprocess.from_audio(scaled[0], gpu=gpu)

                # Log loudness-scaled audio
                key = f'scaled-{int(ratio * 100):03d}/{i:02d}'
                waveforms[f'{key}-audio'] = scaled[0]

                # Log prosody figure
                if i < promonet.PLOT_EXAMPLES:
                    figures[key] = promonet.plot.from_features(
                        scaled,
                        promonet.preprocess.loudness.band_average(predicted_loudness, 1),
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_ppg,
                        promonet.preprocess.loudness.band_average(scaled_loudness, 1),
                        pitch,
                        periodicity,
                        ppg)

                # Update metrics
                metrics[key.split('/')[0]].update(
                    scaled_loudness,
                    pitch,
                    periodicity,
                    ppg,
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg)

        # 指定されたステップ数で評価を終了
        if evaluation_steps is not None and i + 1 == evaluation_steps:
            break

    # --- 最終的なメトリクスを計算してまとめる ---
    for condition in metrics:
        for key, value in metrics[condition]().items():
            scalars[f'{condition}/{key}'] = value

    # 結果をTensorBoardに書き込む
    torchutil.tensorboard.update(
        directory,
        step,
        figures=figures,
        scalars=scalars,
        audio=waveforms,
        sample_rate=promonet.SAMPLE_RATE)
