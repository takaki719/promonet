import math
import os
from typing import List, Optional, Tuple, Union

import ppgs
import pypar
import torch

import promonet


###############################################################################
# 音声特徴量の編集 (Edit speech features)
###############################################################################


def from_features(
    loudness: torch.Tensor,
    pitch: torch.Tensor,
    periodicity: torch.Tensor,
    ppg: torch.Tensor,
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None,
    stretch_unvoiced: bool = True,
    stretch_silence: bool = True,
    return_grid: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    メモリ上にある音声表現（特徴量）を編集します。これが編集処理の中核となる関数です。

    Arguments
        loudness (torch.Tensor): 編集対象の音量系列。
        pitch (torch.Tensor): 編集対象のピッチ系列。
        periodicity (torch.Tensor): 編集対象の周期性系列。
        ppg (torch.Tensor): 編集対象のPPG (音素事後確率) 系列。
        pitch_shift_cents (Optional[float]): ピッチシフト量（セント単位）。
        time_stretch_ratio (Optional[float]): タイムストレッチの比率。1より大きいと速くなります。
        loudness_scale_db (Optional[float]): 音量スケール変更量（dB単位）。(非推奨: `loudness`自体を直接編集する方が望ましい)。
        stretch_unvoiced (bool): Trueの場合、無声のフレームもタイムストレッチの対象にします。
        stretch_silence (bool): Trueの場合、無音のフレームもタイムストレッチの対象にします。
        return_grid (bool): Trueの場合、タイムストレッチに使用したサンプリンググリッドも返します。

    Returns
        Union[Tuple, Tuple]:
            編集後の (loudness, pitch, periodicity, ppg) のタプル。
            `return_grid`がTrueの場合は、上記に加えて `grid` も返します。
    """
    # --- タイムストレッチ (オプション) ---
    if time_stretch_ratio is not None:

        # --- タイムストレッチ用のサンプリンググリッドを作成 ---
        # サンプリンググリッドは、出力の各フレームが入力のどの時点からサンプリングされるべきかを示します。
        
        # もし無声音と無音の両方をストレッチする場合、単純に全体を一様に伸縮させます。
        if stretch_unvoiced and stretch_silence:
            grid = promonet.edit.grid.constant(
                ppg,
                time_stretch_ratio)
        # 有声音のみストレッチするなど、部分的に伸縮させる場合
        else:
            # --- ストレッチ対象の音素を選択 ---
            # まず、有声音の音素インデックスを取得します。
            indices = [
                ppgs.PHONEME_TO_INDEX_MAPPING[phoneme]
                for phoneme in ppgs.VOICED]

            # 無音区間もストレッチ対象に含める場合
            if stretch_silence:
                indices.append(ppgs.PHONEME_TO_INDEX_MAPPING[pypar.SILENCE])

            # 無声音もストレッチ対象に含める場合
            if stretch_unvoiced:
                # 全ての音素から、有声音と無音を除いた集合（＝無声音）のインデックスを追加します。
                indices.extend(
                    list(
                        set(range(len(ppgs.PHONEMES))) -  # 全音素のインデックス
                        set([ppgs.PHONEME_TO_INDEX_MAPPING[p] for p in ppgs.VOICED]) - # 有声音のインデックス
                        set([ppgs.PHONEME_TO_INDEX_MAPPING[pypar.SILENCE]]) # 無音のインデックス
                    )
                )

            # --- 時間に応じて変化するストレッチ比率を計算 ---
            # 各時間フレームが、選択された（ストレッチ対象の）音素である確率を計算します。
            selected = ppg[torch.tensor(indices)].sum(dim=0)

            # 出力されるべき目標フレーム数を計算します。
            target_frames = round(ppg.shape[-1] / time_stretch_ratio)

            # ストレッチ対象の区間だけに適用する「実効的な」ストレッチ比率を計算します。
            # (目標フレーム数 - 非対象フレーム数) / 対象フレーム数
            total_selected = selected.sum()
            total_unselected = ppg.shape[-1] - total_selected
            effective_ratio = (target_frames - total_unselected) / total_selected

            # --- サンプリンググリッドを生成 ---
            # このグリッドは、時間に応じて伸縮率が変わることを表現します。
            grid = torch.zeros(target_frames)
            i = 0.  # 入力フレームにおける現在の位置（浮動小数点）
            # 出力フレームを1つずつ生成していきます。
            for j in range(1, target_frames):

                # 現在位置(i)における、ストレッチ対象である確率を線形補間で求めます。
                left = math.floor(i)
                if left + 1 < len(selected):
                    offset = i - left
                    probability = (
                        offset * selected[left + 1] +
                        (1 - offset) * selected[left])
                else:
                    probability = selected[left]

                # 確率に基づいて、この時点でのストレッチ比率を決定します。
                # 確率が1ならeffective_ratio, 0なら1 (伸縮しない) となります。
                ratio = probability * effective_ratio + (1 - probability)
                step = 1. / ratio # 次の出力フレームを作るために、入力フレームをどれだけ進めるか

                # グリッドを更新し、入力フレームの位置を進めます。
                grid[j] = grid[j - 1] + step
                i += step

        # --- グリッドを使って各特徴量をリサンプリング（タイムストレッチ） ---
        # ピッチは対数領域でリサンプリングしてから元に戻すことで、音楽的に自然な補間を行います。
        pitch = 2 ** promonet.edit.grid.sample(torch.log2(pitch), grid)
        periodicity = promonet.edit.grid.sample(periodicity, grid)
        loudness = promonet.edit.grid.sample(loudness, grid)
        # PPGは専用の補間方法を使います。
        ppg = promonet.edit.grid.sample(ppg, grid, promonet.PPG_INTERP_METHOD)

    # `return_grid`がTrueで、タイムストレッチが行われなかった場合のためにgridをNoneで初期化
    elif return_grid:
        grid = None

    # --- ピッチシフト (オプション) ---
    if pitch_shift_cents is not None:
        # セント単位を周波数比に変換し、ピッチに乗算します。
        pitch = pitch.clone() * promonet.convert.cents_to_ratio(
            pitch_shift_cents)
        # モデルが扱えるピッチの範囲内に収まるようにクリッピングします。
        pitch = torch.clip(pitch, promonet.FMIN, promonet.FMAX)

    # --- 音量スケーリング (オプション) ---
    if loudness_scale_db is not None:
        # dBスケールなので、単純に加算します。
        loudness += loudness_scale_db

    if return_grid:
        return loudness, pitch, periodicity, ppg, grid
    return loudness, pitch, periodicity, ppg


def from_file(
    loudness_file: Union[str, bytes, os.PathLike],
    pitch_file: Union[str, bytes, os.PathLike],
    periodicity_file: Union[str, bytes, os.PathLike],
    ppg_file: Union[str, bytes, os.PathLike],
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None,
    stretch_unvoiced: bool = True,
    stretch_silence: bool = True,
    return_grid: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    ディスク上の特徴量ファイルを読み込んで編集します。

    (引数は from_features とほぼ同じですが、対象がファイルパスになります)
    """
    # 特徴量ファイルをPyTorchテンソルとして読み込みます。
    pitch = torch.load(pitch_file)
    loudness = torch.load(loudness_file)
    periodicity = torch.load(periodicity_file)
    # PPGはピッチの長さに合わせて読み込みます。
    ppg = promonet.load.ppg(ppg_file, pitch.shape[-1])
    
    # メモリに読み込んだ特徴量を使って、中心的な編集関数 from_features を呼び出します。
    return from_features(
        loudness,
        pitch,
        periodicity,
        ppg,
        pitch_shift_cents,
        time_stretch_ratio,
        loudness_scale_db,
        stretch_unvoiced,
        stretch_silence,
        return_grid)


def from_file_to_file(
    loudness_file: Union[str, bytes, os.PathLike],
    pitch_file: Union[str, bytes, os.PathLike],
    periodicity_file: Union[str, bytes, os.PathLike],
    ppg_file: Union[str, bytes, os.PathLike],
    output_prefix: Union[str, bytes, os.PathLike],
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None,
    stretch_unvoiced: bool = True,
    stretch_silence: bool = True,
    save_grid: bool = False
) -> None:
    """
    ディスク上の特徴量ファイルを編集し、結果をディスクに保存します。

    Arguments
        (前半の引数は from_file と同じ)
        output_prefix (str): 出力ファイルの接頭辞（拡張子なし）。
        save_grid (bool): Trueの場合、タイムストレッチのグリッドもファイルに保存します。
    """
    # 1. ファイルを読み込んで編集処理を実行します。
    results = from_file(
        loudness_file,
        pitch_file,
        periodicity_file,
        ppg_file,
        pitch_shift_cents,
        time_stretch_ratio,
        loudness_scale_db,
        stretch_unvoiced,
        stretch_silence,
        save_grid) # グリッドを保存する場合はTrueを渡して受け取る

    # 2. 編集結果をファイルに保存します。
    # ビタビ復号を使った場合はファイル名に'-viterbi'を追加します。
    viterbi = '-viterbi' if promonet.VITERBI_DECODE_PITCH else ''
    torch.save(results[0], f'{output_prefix}-loudness.pt')
    torch.save(results[1], f'{output_prefix}{viterbi}-pitch.pt')
    torch.save(results[2], f'{output_prefix}{viterbi}-periodicity.pt')
    torch.save(results[3], f'{output_prefix}{ppgs.representation_file_extension()}')
    if save_grid:
        torch.save(results[4], f'{output_prefix}-grid.pt')


def from_files_to_files(
    loudness_files: List[Union[str, bytes, os.PathLike]],
    pitch_files: List[Union[str, bytes, os.PathLike]],
    periodicity_files: List[Union[str, bytes, os.PathLike]],
    ppg_files: List[Union[str, bytes, os.PathLike]],
    output_prefixes: List[Union[str, bytes, os.PathLike]],
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None,
    stretch_unvoiced: bool = True,
    stretch_silence: bool = True,
    save_grid: bool = False
) -> None:
    """
    複数の特徴量ファイルを一括で編集し、結果をディスクに保存します。

    (引数は from_file_to_file とほぼ同じですが、対象がファイルパスのリストになります)
    """
    # 入力ファイルと出力接頭辞のリストをまとめてループ処理します。
    for loudness_file, pitch_file, periodicity_file, ppg_file, prefix in zip(
        loudness_files,
        pitch_files,
        periodicity_files,
        ppg_files,
        output_prefixes
    ):
        # 各ファイルセットに対して、単一ファイル用の編集・保存関数を呼び出します。
        from_file_to_file(
            loudness_file,
            pitch_file,
            periodicity_file,
            ppg_file,
            prefix,
            pitch_shift_cents,
            time_stretch_ratio,
            loudness_scale_db,
            stretch_unvoiced,
            stretch_silence,
            save_grid)
