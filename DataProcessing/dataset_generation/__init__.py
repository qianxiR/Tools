# -*- coding: utf-8 -*-
"""
图文数据集生成工具集

提供基于 RemoteCLIP + VLM/LLM 的变化检测（CD）与语义分割（SEG）数据集生成流水线，
包含区域提取、零样本分类、VLM 修正、描述生成、QA 生成、MCI/COCO 格式转换。

入口：python -m Tools.DataProcessing.dataset_generation.main --help
"""

__all__ = ["main"]
