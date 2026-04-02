"""GAN 训练器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from .datasets import ROOT_DIR
from .gan_datasets import create_gan_dataloader
from .gan_advanced_models import ConditionalResGenerator, ProjectionDiscriminator
from .gan_utils import (
    ensure_output_dir,
    get_device,
    save_fake_image_grid,
    save_history_csv,
    save_json,
    save_loss_curve,
    save_model_checkpoint,
    save_real_fake_comparison,
    set_seed,
)


@dataclass
class GANTrainConfig:
    """GAN 训练配置。"""

    image_size: int = 128
    latent_dim: int = 256
    generator_feature_maps: int = 96
    discriminator_feature_maps: int = 96
    batch_size: int = 256
    epochs: int = 80
    learning_rate: float = 5e-5
    beta1: float = 0.5
    beta2: float = 0.999
    num_workers: int = 0
    seed: int = 42
    sample_interval: int = 10
    per_class_limit: int | None = None
    balance_labels: bool = True
    drop_cross_split_duplicates: bool = True
    output_dir: str = "gan——results"
    max_batches: int | None = None
    discriminator_steps: int = 1
    grad_clip_norm: float = 5.0
    use_mixed_precision: bool = False
    defect_multiplier: float = 1.0


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """判别器 hinge loss。"""
    real_loss = torch.relu(1.0 - real_logits).mean()
    fake_loss = torch.relu(1.0 + fake_logits).mean()
    return real_loss + fake_loss


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """生成器 hinge loss。"""
    return -fake_logits.mean()


def train_gan(config: GANTrainConfig) -> Path:
    """执行 GAN 训练并返回输出目录。"""
    set_seed(config.seed)
    device = get_device()
    output_dir = ensure_output_dir(ROOT_DIR / config.output_dir)

    data_bundle = create_gan_dataloader(
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        per_class_limit=config.per_class_limit,
        drop_cross_split_duplicates=config.drop_cross_split_duplicates,
        balance_labels=config.balance_labels,
        defect_multiplier=config.defect_multiplier,
    )

    generator = ConditionalResGenerator(
        latent_dim=config.latent_dim,
        base_channels=config.generator_feature_maps,
        image_size=config.image_size,
        num_classes=2,
    ).to(device)
    discriminator = ProjectionDiscriminator(
        base_channels=config.discriminator_feature_maps,
        image_size=config.image_size,
        num_classes=2,
    ).to(device)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )

    fixed_noise = torch.randn(16, config.latent_dim, 1, 1, device=device)
    fixed_labels = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long, device=device)
    history: list[dict] = []
    latest_real_batch = None
    latest_labels_batch = None
    use_amp = device.type == "cuda" and config.use_mixed_precision
    grad_scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"当前训练设备：{device}")
    print(f"GAN 输出目录：{output_dir}")
    print(f"GAN 训练样本数：{len(data_bundle.train_df)}")
    print(f"当前缺陷件过采样倍数：{config.defect_multiplier:.2f}")
    print(f"当前 GAN 模型：条件式残差 GAN（谱归一化 + 自注意力 + Hinge Loss）")

    for epoch in range(1, config.epochs + 1):
        generator.train()
        discriminator.train()
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        num_steps = 0

        progress = tqdm(
            data_bundle.train_loader,
            desc=f"正在训练第 {epoch} 轮",
            leave=True,
        )
        for batch_idx, (real_images, labels, _) in enumerate(progress):
            if config.max_batches is not None and batch_idx >= config.max_batches:
                break

            real_images = real_images.to(device)
            labels = labels.to(device)
            latest_real_batch = real_images[:16].detach()
            latest_labels_batch = labels[:16].detach()
            batch_size = real_images.size(0)

            d_loss_value = 0.0
            for _ in range(config.discriminator_steps):
                optimizer_d.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, config.latent_dim, 1, 1, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    fake_images = generator(noise, labels)
                    real_logits = discriminator(real_images, labels)
                    fake_logits = discriminator(fake_images.detach(), labels)
                    d_loss = discriminator_hinge_loss(real_logits, fake_logits)
                if not torch.isfinite(d_loss):
                    print("检测到判别器损失出现非有限值，提前停止当前训练。")
                    return output_dir
                grad_scaler_d.scale(d_loss).backward()
                if config.grad_clip_norm is not None:
                    grad_scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_clip_norm)
                grad_scaler_d.step(optimizer_d)
                grad_scaler_d.update()
                d_loss_value += float(d_loss.item())
            d_loss_value /= max(config.discriminator_steps, 1)

            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, config.latent_dim, 1, 1, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                fake_images = generator(noise, labels)
                gen_logits = discriminator(fake_images, labels)
                g_loss = generator_hinge_loss(gen_logits)
            if not torch.isfinite(g_loss):
                print("检测到生成器损失出现非有限值，提前停止当前训练。")
                return output_dir
            grad_scaler_g.scale(g_loss).backward()
            if config.grad_clip_norm is not None:
                grad_scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip_norm)
            grad_scaler_g.step(optimizer_g)
            grad_scaler_g.update()

            g_loss_sum += float(g_loss.item())
            d_loss_sum += d_loss_value
            num_steps += 1
            progress.set_postfix(
                生成器损失=f"{g_loss.item():.4f}",
                判别器损失=f"{d_loss_value:.4f}",
            )

        avg_g_loss = g_loss_sum / max(num_steps, 1)
        avg_d_loss = d_loss_sum / max(num_steps, 1)
        history.append(
            {
                "epoch": epoch,
                "generator_loss": avg_g_loss,
                "discriminator_loss": avg_d_loss,
            }
        )
        print(
            f"第 {epoch} 轮训练完成 | 平均生成器损失：{avg_g_loss:.4f} | "
            f"平均判别器损失：{avg_d_loss:.4f}"
        )

        if epoch % config.sample_interval == 0 or epoch == 1 or epoch == config.epochs:
            generator.eval()
            with torch.no_grad():
                fake_preview = generator(fixed_noise, fixed_labels)
            save_fake_image_grid(
                fake_images=fake_preview,
                output_path=output_dir / f"fake_samples_epoch_{epoch:04d}.png",
                title=f"第 {epoch} 轮生成样本拼图",
            )
            if latest_real_batch is not None and latest_labels_batch is not None:
                preview_real = latest_real_batch
                preview_fake = generator(
                    fixed_noise[: preview_real.size(0)],
                    fixed_labels[: preview_real.size(0)],
                ).detach()
                save_real_fake_comparison(
                    real_images=preview_real,
                    fake_images=preview_fake,
                    output_path=output_dir / f"real_fake_compare_epoch_{epoch:04d}.png",
                    title=f"第 {epoch} 轮真实/生成样本对比",
                )
            save_model_checkpoint(
                generator,
                output_dir / f"generator_epoch_{epoch:04d}.pt",
                extra_state={"config": asdict(config), "epoch": epoch},
            )
            save_model_checkpoint(
                discriminator,
                output_dir / f"discriminator_epoch_{epoch:04d}.pt",
                extra_state={"config": asdict(config), "epoch": epoch},
            )

    history_df = save_history_csv(history, output_dir / "gan_loss_history.csv")
    save_loss_curve(history_df, output_dir / "loss_curve.png")
    save_json(asdict(config), output_dir / "gan_train_config.json")
    save_model_checkpoint(
        generator,
        output_dir / "generator_final.pt",
        extra_state={"config": asdict(config), "epoch": config.epochs},
    )
    save_model_checkpoint(
        discriminator,
        output_dir / "discriminator_final.pt",
        extra_state={"config": asdict(config), "epoch": config.epochs},
    )
    print(f"GAN 训练完成，结果已保存到：{output_dir}")
    return output_dir
