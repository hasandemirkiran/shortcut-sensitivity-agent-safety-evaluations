"""Pinned upstream assets.

The upstream repository stores JSON in Git LFS. media.githubusercontent.com
serves the LFS object content for a path at a pinned commit, so each entry is
also checked against the object's SHA-256 and size.
"""

from __future__ import annotations

from dataclasses import dataclass


EVAL_COMMIT = "46358fa424a927a895c6c8322f99032c4eb5155e"
TRAIN_COMMIT = "e54c46326353fc8ed286f2d602ac28f616defff3"
REPOSITORY = "MurrayTom/ToolSafe"


@dataclass(frozen=True)
class Asset:
    source_path: str
    destination: str
    sha256: str
    size: int
    commit: str
    kind: str

    @property
    def url(self) -> str:
        return (
            f"https://media.githubusercontent.com/media/{REPOSITORY}/"
            f"{self.commit}/{self.source_path}"
        )


def _asset(
    source_path: str,
    destination: str,
    sha256: str,
    size: int,
    *,
    commit: str = EVAL_COMMIT,
    kind: str = "eval",
) -> Asset:
    return Asset(source_path, destination, sha256, size, commit, kind)


ASSETS = (
    _asset(
        "TS-Bench/agentdojo-traj/banking.json",
        "eval/agentdojo/banking.json",
        "eb4a314845c8236ba1bbcd9855973f59279a70368b6eb31d447a4cd243086029",
        239239,
    ),
    _asset(
        "TS-Bench/agentdojo-traj/slack.json",
        "eval/agentdojo/slack.json",
        "08152de8cf9c39d01d58eb708f3a820a892f493841b975e09654ab904cf8a0d5",
        361979,
    ),
    _asset(
        "TS-Bench/agentdojo-traj/travel.json",
        "eval/agentdojo/travel.json",
        "1935121585fb1a02662e8121fe80e78c6d89b77e0c6035ea2b75b593e3728a22",
        1071254,
    ),
    _asset(
        "TS-Bench/agentdojo-traj/workspace.json",
        "eval/agentdojo/workspace.json",
        "74b20a4fcd7df1c3e39a231e64860aeac8cd57ce3531728b329080abbcc862d0",
        8959525,
    ),
    _asset(
        "TS-Bench/agentharm-traj/benign_steps.json",
        "eval/agentharm/benign_steps.json",
        "f44fce25b0ed795e3c11fadafb8d8c959d2ae4513e6e9f15e154fa4aba9bf020",
        689251,
    ),
    _asset(
        "TS-Bench/agentharm-traj/harmful_steps.json",
        "eval/agentharm/harmful_steps.json",
        "6046b4f4b2e5086c98780433bddf74be542eeec910e3260b4727bf22a9a3365b",
        1764105,
    ),
    _asset(
        "TS-Bench/asb-traj/test/DPI_attack_success.json",
        "eval/asb/DPI_attack_success.json",
        "a900a0775addf35588154d28438127f9c830a08845a105e084b319b1e984d4af",
        5205776,
    ),
    _asset(
        "TS-Bench/asb-traj/test/OPI_attack_success.json",
        "eval/asb/OPI_attack_success.json",
        "e491c96e43c8352e65ca948a72cb3e6f366d60f14fd7d3d6e374865eb092ae47",
        4712262,
    ),
    _asset(
        "TS-Bench/asb-traj/test/atttack_failure.json",
        "eval/asb/atttack_failure.json",
        "6cf422389a41416e05d236252ecba8757c1f7d7a978ae270e419ff846ae98f51",
        945992,
    ),
    _asset(
        "guardian_test_logs/agentdojo/TS-Guard/labels.json",
        "reference/agentdojo/labels.json",
        "8b589933b477d9eb86d69e2d1c1c915e039412fb61bf56edd075928c7438e817",
        10982,
        kind="reference",
    ),
    _asset(
        "guardian_test_logs/agentdojo/TS-Guard/preds.json",
        "reference/agentdojo/preds.json",
        "e0f4a79416ae1a510bcdc5620b924848390bb65dce71a944f969e088501682f7",
        10982,
        kind="reference",
    ),
    _asset(
        "guardian_test_logs/agentdojo/TS-Guard/metrics_strict.json",
        "reference/agentdojo/metrics_strict.json",
        "19b580efba2dcb5eaf9cfc3139aae90cab46527ecfba411e7f7fbe2e6af7a95c",
        121,
        kind="reference",
    ),
    _asset(
        "guardian_test_logs/agentharm/TS-Guard/labels.json",
        "reference/agentharm/labels.json",
        "51b127b836995f3780ed42885079a8ecce725709f46f89c860cd6c7a21d67cf5",
        6581,
        kind="reference",
    ),
    _asset(
        "guardian_test_logs/agentharm/TS-Guard/preds.json",
        "reference/agentharm/preds.json",
        "fb21f86e34bc64f3c340c731980e07b05ebec48ea55058f34ea75ae826b00a32",
        6581,
        kind="reference",
    ),
    _asset(
        "guardian_test_logs/agentharm/TS-Guard/metrics_strict.json",
        "reference/agentharm/metrics_strict.json",
        "aa69da0249a41504ac530cc677c782e8c5b33f15907cf8231be8d0bcae655c4e",
        120,
        kind="reference",
    ),
    _asset(
        "guardian_test_logs/asb/all/TS-Guard/labels.json",
        "reference/asb/labels.json",
        "e8f73b68c3f7017fdd6eab8a03bb8e828011449f34b7a11d08476371306a72ba",
        39551,
        kind="reference",
    ),
    _asset(
        "guardian_test_logs/asb/all/TS-Guard/preds.json",
        "reference/asb/preds.json",
        "1aad09546b09a388344099aa79b93b7b1ea69e00c6ed7d1f3371025c0be858b9",
        47081,
        kind="reference",
    ),
    _asset(
        "guardian_test_logs/asb/all/TS-Guard/metrics_strict.json",
        "reference/asb/metrics_strict.json",
        "66cb4c8f2b927f619c6077e8469a535f2fb14aa35b58ca9556fdfca345eb3f6f",
        121,
        kind="reference",
    ),
    _asset(
        "TS-Guard/verl-main/data/agentsafety/train_agentalign-harm-seedset1.json",
        "train/train_agentalign-harm-seedset1.json",
        "f77dbb9604978b9c918d8bad85f99ba0a96fec89675d2a7cdb5edc05ce24e458",
        1625720,
        commit=TRAIN_COMMIT,
        kind="train",
    ),
    _asset(
        "TS-Guard/verl-main/data/agentsafety/train_agentalign-harm-seedset0.json",
        "train/train_agentalign-harm-seedset0.json",
        "e7eff71fbe8c77c9f08ba9a8ee65e7e591c6d87a8158b27feb5b0599e34fdd60",
        3063725,
        commit=TRAIN_COMMIT,
        kind="train",
    ),
    _asset(
        "TS-Guard/verl-main/data/agentsafety/train_agentalign-benign-seedset0.json",
        "train/train_agentalign-benign-seedset0.json",
        "598288520812c6275931e723dffb74a5a2e03cf6adaa66038687458676ec7e4e",
        799405,
        commit=TRAIN_COMMIT,
        kind="train",
    ),
    _asset(
        "TS-Guard/verl-main/data/agentsafety/train_asb-dpi_v2.json",
        "train/train_asb-dpi_v2.json",
        "0bcbf55251941b775cab14ec0a24a36cb67bb035cdeb0235bcc6b052b6fd978f",
        4817552,
        commit=TRAIN_COMMIT,
        kind="train",
    ),
    _asset(
        "TS-Guard/verl-main/data/agentsafety/train_asb-opi.json",
        "train/train_asb-opi.json",
        "ac7149abba410a8d6398727aad2ec72381857cc7352897d2ff711aa5071432a3",
        5981955,
        commit=TRAIN_COMMIT,
        kind="train",
    ),
    _asset(
        "TS-Guard/verl-main/data/agentsafety/val_agentalign-harm-seedset1.json",
        "validation/val_agentalign-harm-seedset1.json",
        "5e6aada8841691cb7c26bd610e533b4b9e01c91a8be588fa95add6240441bccb",
        1567353,
        commit=TRAIN_COMMIT,
        kind="validation",
    ),
    _asset(
        "TS-Guard/verl-main/data/agentsafety/val_agentalign-harm-seedset0.json",
        "validation/val_agentalign-harm-seedset0.json",
        "f76465488fadfb0ba455c831dd26b1d0d59e249d549f0f7799e9034eeec36771",
        682157,
        commit=TRAIN_COMMIT,
        kind="validation",
    ),
    _asset(
        "TS-Guard/verl-main/data/agentsafety/val_agentalign-benign-seedset0.json",
        "validation/val_agentalign-benign-seedset0.json",
        "14543db5ac1e346bd7dd6f723c3b99a36db075118ac73f26e88253766cbbcb26",
        767014,
        commit=TRAIN_COMMIT,
        kind="validation",
    ),
    _asset(
        "TS-Guard/verl-main/data/agentsafety/val_asb-dpi_v2.json",
        "validation/val_asb-dpi_v2.json",
        "172adf053ff63d2ea40a24f575d27f396f23bdc7efe7820ea8e217d3a41f149b",
        1220039,
        commit=TRAIN_COMMIT,
        kind="validation",
    ),
    _asset(
        "TS-Guard/verl-main/data/agentsafety/val_asb-opi.json",
        "validation/val_asb-opi.json",
        "a61f095f2b6ad033f43c810241ffa6f46e13d1f048ce63f9afbc9684d8aa4c93",
        1473592,
        commit=TRAIN_COMMIT,
        kind="validation",
    ),
)

